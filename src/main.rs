use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
    fs,
    io::{self, Stdout},
    path::{Path, PathBuf},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use clap::{Arg, ArgAction, Command};
use crossbeam_channel as channel;
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
        MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use foldhash::fast::RandomState as FastHash;
use ignore::{WalkBuilder, WalkState};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Cell, Clear, Paragraph, Row, Scrollbar, ScrollbarOrientation,
        ScrollbarState, Table, TableState, Wrap,
    },
};
use rayon::prelude::*;

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

struct Args {
    path: Option<PathBuf>,
}

impl Args {
    fn parse() -> Self {
        let matches = cli_command().get_matches();
        Self {
            path: matches.get_one::<PathBuf>("path").cloned(),
        }
    }
}

fn cli_command() -> Command {
    let command = Command::new(env!("CARGO_PKG_NAME"))
        .version(env!("CARGO_PKG_VERSION"))
        .disable_version_flag(true);
    let command = match env!("CARGO_PKG_DESCRIPTION") {
        "" => command,
        description => command.about(description),
    };

    command
        .arg(
            Arg::new("path")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf))
                .help("Directory or file to inspect. Defaults to the current directory"),
        )
        .arg(
            Arg::new("version")
                .short('v')
                .long("version")
                .action(ArgAction::Version)
                .help("Print version"),
        )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeKind {
    Directory,
    File,
    Symlink,
    Other,
}

impl NodeKind {
    fn label(self) -> &'static str {
        match self {
            Self::Directory => "dir",
            Self::File => "file",
            Self::Symlink => "link",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone)]
struct EntryRecord {
    path: Arc<Path>,
    kind: NodeKind,
    own_size: u64,
}

#[derive(Debug, Clone)]
struct Node {
    path: Arc<Path>,
    name: String,
    parent: Option<usize>,
    children: Vec<usize>,
    kind: NodeKind,
    own_size: u64,
    total_size: u64,
    subtree_entries: usize,
}

#[derive(Debug)]
struct Inventory {
    root: PathBuf,
    nodes: Vec<Node>,
    path_index: PathIndex,
    scanned_entries: usize,
    errors: Vec<String>,
    elapsed: Duration,
}

type PathIndex = HashMap<Arc<Path>, usize, FastHash>;

#[derive(Debug, Clone)]
struct DeleteTarget {
    path: PathBuf,
    display_name: String,
    total_size: u64,
    kind: NodeKind,
}

#[derive(Debug, Clone)]
struct DeletePlan {
    targets: Vec<DeleteTarget>,
    total_size: u64,
}

impl DeletePlan {
    fn from_targets(targets: Vec<DeleteTarget>) -> Option<Self> {
        let targets = prune_nested_targets(targets);
        if targets.is_empty() {
            return None;
        }

        let total_size = targets.iter().fold(0u64, |total, target| {
            total.saturating_add(target.total_size)
        });
        Some(Self {
            targets,
            total_size,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfirmChoice {
    Cancel,
    Delete,
}

#[derive(Debug, Clone)]
struct DeleteConfirmation {
    plan: DeletePlan,
    choice: ConfirmChoice,
}

impl DeleteConfirmation {
    fn new(plan: DeletePlan) -> Self {
        Self {
            plan,
            choice: ConfirmChoice::Cancel,
        }
    }

    fn choose_cancel(&mut self) {
        self.choice = ConfirmChoice::Cancel;
    }

    fn choose_delete(&mut self) {
        self.choice = ConfirmChoice::Delete;
    }

    fn toggle(&mut self) {
        self.choice = match self.choice {
            ConfirmChoice::Cancel => ConfirmChoice::Delete,
            ConfirmChoice::Delete => ConfirmChoice::Cancel,
        };
    }
}

#[derive(Debug)]
struct DeleteStats {
    removed_files: usize,
    removed_dirs: usize,
    deleted_targets: Vec<PathBuf>,
    errors: Vec<String>,
    elapsed: Duration,
}

#[derive(Debug)]
struct App {
    inventory: Inventory,
    current: usize,
    selected: usize,
    scroll: usize,
    visible_rows: usize,
    marked: HashSet<usize>,
    history: Vec<ViewState>,
    inline_scan: Option<InlineScanStatus>,
    threads: usize,
    one_file_system: bool,
    confirm: Option<DeleteConfirmation>,
    status: String,
    should_quit: bool,
}

#[derive(Debug, Clone, Copy)]
struct ViewState {
    current: usize,
    selected: usize,
    scroll: usize,
}

#[derive(Debug, Clone)]
struct InlineScanStatus {
    path: PathBuf,
    started: Instant,
    progress: ScanProgress,
}

#[derive(Debug)]
enum AppCommand {
    None,
    Delete(DeletePlan),
    Rescan,
    ScanParent(ParentScanRequest),
}

#[derive(Debug)]
struct ParentScanRequest {
    parent_path: PathBuf,
    reuse: ReusableSubtree,
    select_path: PathBuf,
}

#[derive(Debug)]
struct ReusableSubtree {
    root: PathBuf,
    records: Vec<EntryRecord>,
    errors: Vec<String>,
}

struct ParentScanTask {
    progress_rx: channel::Receiver<ScanProgress>,
    result_rx: channel::Receiver<Result<Inventory>>,
    handle: thread::JoinHandle<()>,
    select_path: PathBuf,
}

impl ReusableSubtree {
    fn from_inventory(inventory: &Inventory) -> Self {
        let records = inventory
            .nodes
            .iter()
            .map(|node| EntryRecord {
                path: node.path.clone(),
                kind: node.kind,
                own_size: node.own_size,
            })
            .collect();

        Self {
            root: inventory.root.clone(),
            records,
            errors: inventory.errors.clone(),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let threads = default_thread_count();
    let one_file_system = false;
    let root = args
        .path
        .unwrap_or(std::env::current_dir().context("failed to read current directory")?);

    let mut tui = Tui::enter()?;
    let inventory = scan_with_progress_screen(&mut tui, root, threads, one_file_system)?;
    let Some(inventory) = inventory else {
        return Ok(());
    };
    run_tui(&mut tui, inventory, threads, one_file_system)
}

fn default_thread_count() -> usize {
    thread::available_parallelism()
        .map(|count| count.get().saturating_mul(2))
        .unwrap_or(8)
        .max(4)
}

fn normalize_root(path: PathBuf) -> Result<PathBuf> {
    fs::canonicalize(&path).with_context(|| format!("failed to resolve path {}", path.display()))
}

fn scan_inventory_with_reuse(
    path: PathBuf,
    threads: usize,
    one_file_system: bool,
    reuse: Option<ReusableSubtree>,
    progress_tx: Option<channel::Sender<ScanProgress>>,
) -> Result<Inventory> {
    let root = normalize_root(path)?;
    let start = Instant::now();
    let root_metadata = fs::symlink_metadata(&root)
        .with_context(|| format!("failed to read metadata for {}", root.display()))?;
    let root_record = record_from_metadata(Arc::from(root.as_path()), root_metadata);

    let mut records = Vec::with_capacity(32_768);
    records.push(root_record);
    let mut errors = Vec::new();
    let skip_path = reuse
        .as_ref()
        .filter(|reuse| reuse.root != root && reuse.root.starts_with(&root))
        .map(|reuse| reuse.root.clone());
    if skip_path.is_some() {
        if let Some(reuse) = reuse {
            records.extend(reuse.records);
            errors.extend(reuse.errors);
        }
    }
    report_scan_progress(&progress_tx, records.len(), errors.len());

    if records[0].kind == NodeKind::Directory {
        let (tx, rx) = channel::unbounded::<ScanMessage>();
        let walker_root = root.clone();
        let skip_path = skip_path.clone();
        let walker = thread::spawn(move || {
            let mut builder = WalkBuilder::new(&walker_root);
            builder
                .hidden(false)
                .ignore(false)
                .parents(false)
                .git_ignore(false)
                .git_global(false)
                .git_exclude(false)
                .follow_links(false)
                .same_file_system(one_file_system)
                .threads(threads);

            builder.build_parallel().run(|| {
                let tx = tx.clone();
                let root = walker_root.clone();
                let skip_path = skip_path.clone();
                let mut batch = RecordBatch::new(tx.clone());
                Box::new(move |result| {
                    match result {
                        Ok(entry) => {
                            let path = entry.path();
                            if skip_path.as_deref() == Some(path) {
                                return WalkState::Skip;
                            }
                            if path != root.as_path() {
                                match fs::symlink_metadata(path) {
                                    Ok(metadata) => {
                                        batch.push(record_from_metadata(Arc::from(path), metadata));
                                    }
                                    Err(error) => {
                                        batch.error(format!("{}: {error}", path.display()));
                                    }
                                }
                            }
                        }
                        Err(error) => {
                            batch.error(error.to_string());
                        }
                    }
                    WalkState::Continue
                })
            });
        });

        let mut entries_since_report = 0usize;
        let mut last_reported_errors = errors.len();
        for message in rx {
            match message {
                ScanMessage::Batch(batch) => {
                    entries_since_report = entries_since_report.saturating_add(batch.len());
                    records.extend(batch);
                }
                ScanMessage::Error(error) => errors.push(error),
            }
            if entries_since_report >= 256 || errors.len() != last_reported_errors {
                report_scan_progress(&progress_tx, records.len(), errors.len());
                entries_since_report = 0;
                last_reported_errors = errors.len();
            }
        }

        walker
            .join()
            .map_err(|_| anyhow!("scanner worker thread panicked"))?;
    }

    let scanned_entries = records.len();
    report_scan_progress(&progress_tx, scanned_entries, errors.len());
    let mut inventory = build_inventory(root, records);
    inventory.scanned_entries = scanned_entries;
    inventory.errors = errors;
    inventory.elapsed = start.elapsed();
    report_scan_progress(
        &progress_tx,
        inventory.scanned_entries,
        inventory.errors.len(),
    );
    Ok(inventory)
}

#[derive(Debug, Clone, Copy, Default)]
struct ScanProgress {
    scanned_entries: usize,
    errors: usize,
}

fn report_scan_progress(
    progress_tx: &Option<channel::Sender<ScanProgress>>,
    scanned_entries: usize,
    errors: usize,
) {
    if let Some(tx) = progress_tx {
        let _ = tx.send(ScanProgress {
            scanned_entries,
            errors,
        });
    }
}

#[cfg(test)]
fn scan_inventory(path: PathBuf, threads: usize, one_file_system: bool) -> Result<Inventory> {
    scan_inventory_with_reuse(path, threads, one_file_system, None, None)
}

#[derive(Debug)]
enum ScanMessage {
    Batch(Vec<EntryRecord>),
    Error(String),
}

/// Per-walker-thread buffer that batches records before sending them over the
/// channel. Sending one message per entry is a measurable cost on large scans;
/// batching cuts channel traffic by orders of magnitude. `Drop` flushes whatever
/// is left when the walker finishes.
struct RecordBatch {
    tx: channel::Sender<ScanMessage>,
    records: Vec<EntryRecord>,
}

const RECORD_BATCH_SIZE: usize = 1024;

impl RecordBatch {
    fn new(tx: channel::Sender<ScanMessage>) -> Self {
        Self {
            tx,
            records: Vec::with_capacity(RECORD_BATCH_SIZE),
        }
    }

    fn push(&mut self, record: EntryRecord) {
        self.records.push(record);
        if self.records.len() >= RECORD_BATCH_SIZE {
            self.flush();
        }
    }

    fn error(&self, error: String) {
        let _ = self.tx.send(ScanMessage::Error(error));
    }

    fn flush(&mut self) {
        if self.records.is_empty() {
            return;
        }
        let batch = std::mem::take(&mut self.records);
        self.records = Vec::with_capacity(RECORD_BATCH_SIZE);
        let _ = self.tx.send(ScanMessage::Batch(batch));
    }
}

impl Drop for RecordBatch {
    fn drop(&mut self) {
        self.flush();
    }
}

fn record_from_metadata(path: Arc<Path>, metadata: fs::Metadata) -> EntryRecord {
    let file_type = metadata.file_type();
    let kind = if file_type.is_dir() {
        NodeKind::Directory
    } else if file_type.is_file() {
        NodeKind::File
    } else if file_type.is_symlink() {
        NodeKind::Symlink
    } else {
        NodeKind::Other
    };
    let own_size = disk_size(&metadata);

    EntryRecord {
        path,
        kind,
        own_size,
    }
}

#[cfg(unix)]
fn disk_size(metadata: &fs::Metadata) -> u64 {
    metadata.blocks().saturating_mul(512)
}

#[cfg(not(unix))]
fn disk_size(metadata: &fs::Metadata) -> u64 {
    metadata.len()
}

fn build_inventory(root: PathBuf, records: Vec<EntryRecord>) -> Inventory {
    // The root record is always inserted first (before any reused records), and the
    // walker never emits a second record for it, so it is always at index 0.
    let mut records = records.into_iter();
    let capacity = records.len() + 1;
    let root_record = records
        .next()
        .expect("root record is inserted before build");

    let root_name = root
        .file_name()
        .map(display_os_str)
        .unwrap_or_else(|| root.display().to_string());

    let mut nodes = Vec::with_capacity(capacity);
    let mut path_index = PathIndex::with_capacity_and_hasher(capacity, FastHash::default());
    let root_path: Arc<Path> = Arc::from(root.as_path());
    nodes.push(Node {
        path: root_path.clone(),
        name: root_name,
        parent: None,
        children: Vec::new(),
        kind: root_record.kind,
        own_size: root_record.own_size,
        total_size: root_record.own_size,
        subtree_entries: 0,
    });
    path_index.insert(root_path, 0);

    // Records arrive in walker order, not sorted, but `ensure_directory_node` creates
    // any missing ancestors before their descendants. That preserves the invariant that
    // a parent's node id is always lower than its children's ids, which the roll-up pass
    // below relies on.
    for record in records {
        let parent_path = record.path.parent().unwrap_or(&root);
        let parent = ensure_directory_node(parent_path, &mut nodes, &mut path_index);
        // Only directories can already exist here: they may have been created as an
        // ancestor placeholder while inserting a descendant that the walker yielded
        // first. Files and symlinks are always unique leaves.
        if record.kind == NodeKind::Directory
            && let Some(&id) = path_index.get(record.path.as_ref())
        {
            nodes[id].kind = record.kind;
            nodes[id].own_size = record.own_size;
            continue;
        }

        let id = nodes.len();
        let name = record
            .path
            .file_name()
            .map(display_os_str)
            .unwrap_or_else(|| record.path.display().to_string());
        path_index.insert(record.path.clone(), id);
        nodes.push(Node {
            path: record.path,
            name,
            parent: Some(parent),
            children: Vec::new(),
            kind: record.kind,
            own_size: record.own_size,
            total_size: record.own_size,
            subtree_entries: 0,
        });
        nodes[parent].children.push(id);
    }

    for id in (0..nodes.len()).rev() {
        let mut total_size = nodes[id].own_size;
        let mut subtree_entries = 0usize;
        for child in &nodes[id].children {
            total_size = total_size.saturating_add(nodes[*child].total_size);
            subtree_entries = subtree_entries.saturating_add(1 + nodes[*child].subtree_entries);
        }
        nodes[id].total_size = total_size;
        nodes[id].subtree_entries = subtree_entries;

        if nodes[id].children.len() > 1 {
            // Take the child list out so the comparator can borrow the other nodes.
            let mut children = std::mem::take(&mut nodes[id].children);
            children.sort_unstable_by(|left, right| {
                nodes[*right]
                    .total_size
                    .cmp(&nodes[*left].total_size)
                    .then_with(|| nodes[*left].name.cmp(&nodes[*right].name))
            });
            nodes[id].children = children;
        }
    }

    Inventory {
        root,
        nodes,
        path_index,
        scanned_entries: 0,
        errors: Vec::new(),
        elapsed: Duration::ZERO,
    }
}

fn ensure_directory_node(path: &Path, nodes: &mut Vec<Node>, path_index: &mut PathIndex) -> usize {
    if let Some(&id) = path_index.get(path) {
        return id;
    }

    // Collect the missing ancestor chain. Every entry is a sub-slice of `path`,
    // so this does not allocate any path strings.
    let mut missing: Vec<&Path> = Vec::with_capacity(2);
    let mut current = path;
    loop {
        if path_index.contains_key(current) {
            break;
        }
        missing.push(current);
        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }

    let mut parent_id = path_index.get(current).copied().unwrap_or(0);
    for missing_path in missing.into_iter().rev() {
        let id = nodes.len();
        let name = missing_path
            .file_name()
            .map(display_os_str)
            .unwrap_or_else(|| missing_path.display().to_string());
        let owned: Arc<Path> = Arc::from(missing_path);
        nodes.push(Node {
            path: owned.clone(),
            name,
            parent: Some(parent_id),
            children: Vec::new(),
            kind: NodeKind::Directory,
            own_size: 0,
            total_size: 0,
            subtree_entries: 0,
        });
        nodes[parent_id].children.push(id);
        path_index.insert(owned, id);
        parent_id = id;
    }

    parent_id
}

fn path_depth(path: &Path) -> usize {
    path.components().count()
}

fn display_os_str(value: &std::ffi::OsStr) -> String {
    value.to_string_lossy().into_owned()
}

fn fast_delete(path: &Path, threads: usize, one_file_system: bool) -> Result<DeleteStats> {
    let start = Instant::now();
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to read metadata for {}", path.display()))?;

    if !metadata.file_type().is_dir() {
        fs::remove_file(path)
            .with_context(|| format!("failed to delete file {}", path.display()))?;
        return Ok(DeleteStats {
            removed_files: 1,
            removed_dirs: 0,
            deleted_targets: Vec::new(),
            errors: Vec::new(),
            elapsed: start.elapsed(),
        });
    }

    let (mut files, mut dirs, mut errors) = collect_delete_entries(path, threads, one_file_system);
    files.sort_unstable();
    files.dedup();
    dirs.sort_unstable();
    dirs.dedup();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .context("failed to create deletion thread pool")?;

    let (file_error_tx, file_error_rx) = channel::unbounded::<String>();
    pool.install(|| {
        files.par_iter().for_each(|file| {
            if let Err(error) = fs::remove_file(file) {
                let _ = file_error_tx.send(format!("{}: {error}", file.display()));
            }
        });
    });
    drop(file_error_tx);
    errors.extend(file_error_rx.iter());

    dirs.retain(|dir| dir != path);
    dirs.sort_unstable_by_key(|dir| Reverse(path_depth(dir)));

    let (dir_error_tx, dir_error_rx) = channel::unbounded::<String>();
    let mut index = 0usize;
    while index < dirs.len() {
        let depth = path_depth(&dirs[index]);
        let mut end = index + 1;
        while end < dirs.len() && path_depth(&dirs[end]) == depth {
            end += 1;
        }
        pool.install(|| {
            dirs[index..end].par_iter().for_each(|dir| {
                if let Err(error) = fs::remove_dir(dir) {
                    let _ = dir_error_tx.send(format!("{}: {error}", dir.display()));
                }
            });
        });
        index = end;
    }
    drop(dir_error_tx);
    errors.extend(dir_error_rx.iter());

    match fs::remove_dir(path) {
        Ok(()) => {}
        Err(error) => errors.push(format!("{}: {error}", path.display())),
    }

    Ok(DeleteStats {
        removed_files: files.len(),
        removed_dirs: dirs.len().saturating_add(1),
        deleted_targets: Vec::new(),
        errors,
        elapsed: start.elapsed(),
    })
}

fn fast_delete_targets(
    targets: &[DeleteTarget],
    threads: usize,
    one_file_system: bool,
) -> DeleteStats {
    let start = Instant::now();
    let mut removed_files = 0usize;
    let mut removed_dirs = 0usize;
    let mut deleted_targets = Vec::new();
    let mut errors = Vec::new();

    for target in targets {
        match fast_delete(&target.path, threads, one_file_system) {
            Ok(stats) => {
                removed_files = removed_files.saturating_add(stats.removed_files);
                removed_dirs = removed_dirs.saturating_add(stats.removed_dirs);
                errors.extend(stats.errors);
                if path_is_gone(&target.path) {
                    deleted_targets.push(target.path.clone());
                }
            }
            Err(error) => {
                errors.push(format!("{}: {error:#}", target.path.display()));
                if path_is_gone(&target.path) {
                    deleted_targets.push(target.path.clone());
                }
            }
        }
    }

    DeleteStats {
        removed_files,
        removed_dirs,
        deleted_targets,
        errors,
        elapsed: start.elapsed(),
    }
}

fn path_is_gone(path: &Path) -> bool {
    match fs::symlink_metadata(path) {
        Ok(_) => false,
        Err(error) => error.kind() == io::ErrorKind::NotFound,
    }
}

fn prune_nested_targets(mut targets: Vec<DeleteTarget>) -> Vec<DeleteTarget> {
    targets.sort_unstable_by(|left, right| {
        path_depth(&left.path)
            .cmp(&path_depth(&right.path))
            .then_with(|| left.path.cmp(&right.path))
    });

    let mut kept: Vec<DeleteTarget> = Vec::with_capacity(targets.len());
    'target: for target in targets {
        for parent in &kept {
            if target.path != parent.path && target.path.starts_with(&parent.path) {
                continue 'target;
            }
        }
        kept.push(target);
    }
    kept
}

fn delete_status(plan: &DeletePlan) -> String {
    if plan.targets.len() == 1 {
        return format!("Deleting {}...", plan.targets[0].path.display());
    }

    format!("Deleting {} selected paths...", plan.targets.len())
}

fn delete_result_status(stats: &DeleteStats, updated_roots: usize) -> String {
    let base = format!(
        "Deleted {} files and {} directories in {}.",
        stats.removed_files,
        stats.removed_dirs,
        format_duration(stats.elapsed)
    );
    let view = if updated_roots == 0 {
        "No confirmed target was removed from the view."
    } else {
        "View updated."
    };

    if stats.errors.is_empty() {
        return format!("{base} {view}");
    }

    let error_word = if stats.errors.len() == 1 {
        "error"
    } else {
        "errors"
    };
    format!(
        "{base} {} {error_word}; {view} Press r to rescan if needed.",
        stats.errors.len()
    )
}

fn collect_delete_entries(
    path: &Path,
    threads: usize,
    one_file_system: bool,
) -> (Vec<PathBuf>, Vec<PathBuf>, Vec<String>) {
    let (tx, rx) = channel::unbounded::<DeleteEntry>();
    let root = path.to_path_buf();
    let walker = thread::spawn(move || {
        let mut builder = WalkBuilder::new(&root);
        builder
            .hidden(false)
            .ignore(false)
            .parents(false)
            .git_ignore(false)
            .git_global(false)
            .git_exclude(false)
            .follow_links(false)
            .same_file_system(one_file_system)
            .threads(threads.max(1));

        builder.build_parallel().run(|| {
            let tx = tx.clone();
            Box::new(move |result| {
                match result {
                    Ok(entry) => {
                        let entry_path = entry.path();
                        match fs::symlink_metadata(entry_path) {
                            Ok(metadata) => {
                                if metadata.file_type().is_dir() {
                                    let _ =
                                        tx.send(DeleteEntry::Directory(entry_path.to_path_buf()));
                                } else {
                                    let _ = tx.send(DeleteEntry::File(entry_path.to_path_buf()));
                                }
                            }
                            Err(error) => {
                                let _ = tx.send(DeleteEntry::Error(format!(
                                    "{}: {error}",
                                    entry_path.display()
                                )));
                            }
                        }
                    }
                    Err(error) => {
                        let _ = tx.send(DeleteEntry::Error(error.to_string()));
                    }
                }
                WalkState::Continue
            })
        });
    });

    let mut files = Vec::new();
    let mut dirs = Vec::new();
    let mut errors = Vec::new();
    for entry in rx {
        match entry {
            DeleteEntry::File(path) => files.push(path),
            DeleteEntry::Directory(path) => dirs.push(path),
            DeleteEntry::Error(error) => errors.push(error),
        }
    }

    if walker.join().is_err() {
        errors.push("delete walker thread panicked".to_string());
    }

    (files, dirs, errors)
}

#[derive(Debug)]
enum DeleteEntry {
    File(PathBuf),
    Directory(PathBuf),
    Error(String),
}

fn scan_with_progress_screen(
    tui: &mut Tui,
    path: PathBuf,
    threads: usize,
    one_file_system: bool,
) -> Result<Option<Inventory>> {
    scan_with_reuse_progress_screen(tui, path, threads, one_file_system, None)
}

fn scan_with_reuse_progress_screen(
    tui: &mut Tui,
    path: PathBuf,
    threads: usize,
    one_file_system: bool,
    reuse: Option<ReusableSubtree>,
) -> Result<Option<Inventory>> {
    let (progress_tx, progress_rx) = channel::unbounded::<ScanProgress>();
    let (result_tx, result_rx) = channel::bounded::<Result<Inventory>>(1);
    let scan_path = path.clone();
    let started = Instant::now();
    let handle = thread::spawn(move || {
        let result = scan_inventory_with_reuse(
            scan_path,
            threads,
            one_file_system,
            reuse,
            Some(progress_tx),
        );
        let _ = result_tx.send(result);
    });

    let mut progress = ScanProgress::default();
    loop {
        while let Ok(update) = progress_rx.try_recv() {
            progress = update;
        }

        match result_rx.try_recv() {
            Ok(result) => {
                handle
                    .join()
                    .map_err(|_| anyhow!("scanner worker thread panicked"))?;
                return result.map(Some);
            }
            Err(channel::TryRecvError::Disconnected) => {
                handle
                    .join()
                    .map_err(|_| anyhow!("scanner worker thread panicked"))?;
                return Err(anyhow!("scanner finished without a result"));
            }
            Err(channel::TryRecvError::Empty) => {}
        }

        tui.terminal
            .draw(|frame| draw_scan_progress(frame, &path, progress, started))?;
        if event::poll(Duration::from_millis(80))? {
            if let Event::Key(key) = event::read()? {
                if is_ctrl_c(key) {
                    return Ok(None);
                }
            }
        }
    }
}

fn run_tui(
    tui: &mut Tui,
    inventory: Inventory,
    threads: usize,
    one_file_system: bool,
) -> Result<()> {
    let mut app = App::new(inventory, threads, one_file_system);
    let mut parent_scan: Option<ParentScanTask> = None;

    loop {
        poll_parent_scan(&mut parent_scan, &mut app)?;
        tui.terminal.draw(|frame| draw(frame, &mut app))?;

        let poll_duration = if parent_scan.is_some() {
            Duration::from_millis(80)
        } else {
            Duration::from_millis(150)
        };

        if event::poll(poll_duration)? {
            let command = match event::read()? {
                Event::Key(key) if parent_scan.is_some() => app.handle_inline_scan_key(key),
                Event::Key(key) => app.handle_key(key),
                Event::Mouse(mouse) => app.handle_mouse(mouse.kind),
                Event::Resize(_, _) => AppCommand::None,
                _ => AppCommand::None,
            };

            match command {
                AppCommand::None => {}
                AppCommand::Rescan => {
                    let current_path = app.inventory.nodes[app.current].path.clone();
                    match scan_with_progress_screen(
                        tui,
                        app.inventory.root.clone(),
                        app.threads,
                        app.one_file_system,
                    ) {
                        Ok(Some(inventory)) => {
                            app.replace_inventory(inventory, &current_path);
                            app.status = "Rescan complete.".to_string();
                        }
                        Ok(None) => {
                            app.should_quit = true;
                        }
                        Err(error) => {
                            app.status = format!("Rescan failed: {error:#}");
                        }
                    }
                }
                AppCommand::ScanParent(request) => {
                    if parent_scan.is_none() {
                        let (task, status) =
                            start_parent_scan(request, app.threads, app.one_file_system);
                        app.inline_scan = Some(status);
                        app.status.clear();
                        parent_scan = Some(task);
                    }
                }
                AppCommand::Delete(plan) => {
                    app.status = delete_status(&plan);
                    tui.terminal.draw(|frame| draw(frame, &mut app))?;

                    let stats =
                        fast_delete_targets(&plan.targets, app.threads, app.one_file_system);
                    let updated_roots = app.remove_deleted_targets(&stats.deleted_targets);
                    app.status = delete_result_status(&stats, updated_roots);
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

fn start_parent_scan(
    request: ParentScanRequest,
    threads: usize,
    one_file_system: bool,
) -> (ParentScanTask, InlineScanStatus) {
    let (progress_tx, progress_rx) = channel::unbounded::<ScanProgress>();
    let (result_tx, result_rx) = channel::bounded::<Result<Inventory>>(1);
    let parent_path = request.parent_path.clone();
    let initial_progress = ScanProgress {
        scanned_entries: request.reuse.records.len(),
        errors: request.reuse.errors.len(),
    };
    let reuse = request.reuse;
    let handle = thread::spawn(move || {
        let result = scan_inventory_with_reuse(
            parent_path,
            threads,
            one_file_system,
            Some(reuse),
            Some(progress_tx),
        );
        let _ = result_tx.send(result);
    });

    let status = InlineScanStatus {
        path: request.parent_path,
        started: Instant::now(),
        progress: initial_progress,
    };
    let task = ParentScanTask {
        progress_rx,
        result_rx,
        handle,
        select_path: request.select_path,
    };

    (task, status)
}

fn poll_parent_scan(task: &mut Option<ParentScanTask>, app: &mut App) -> Result<()> {
    let Some(active) = task.as_mut() else {
        return Ok(());
    };

    while let Ok(progress) = active.progress_rx.try_recv() {
        if let Some(status) = &mut app.inline_scan {
            status.progress = progress;
        }
    }

    match active.result_rx.try_recv() {
        Ok(result) => {
            let active = task.take().expect("active parent scan exists");
            active
                .handle
                .join()
                .map_err(|_| anyhow!("scanner worker thread panicked"))?;
            app.inline_scan = None;
            match result {
                Ok(inventory) => {
                    app.replace_inventory_selecting_child(inventory, &active.select_path);
                    app.status = "Parent directory scanned.".to_string();
                }
                Err(error) => {
                    app.status = format!("Parent scan failed: {error:#}");
                }
            }
        }
        Err(channel::TryRecvError::Disconnected) => {
            let active = task.take().expect("active parent scan exists");
            active
                .handle
                .join()
                .map_err(|_| anyhow!("scanner worker thread panicked"))?;
            app.inline_scan = None;
            app.status = "Parent scan failed: scanner finished without a result.".to_string();
        }
        Err(channel::TryRecvError::Empty) => {}
    }

    Ok(())
}

impl App {
    fn new(inventory: Inventory, threads: usize, one_file_system: bool) -> Self {
        Self {
            inventory,
            current: 0,
            selected: 0,
            scroll: 0,
            visible_rows: 1,
            marked: HashSet::new(),
            history: Vec::new(),
            inline_scan: None,
            threads,
            one_file_system,
            confirm: None,
            status: String::new(),
            should_quit: false,
        }
    }

    fn replace_inventory(&mut self, inventory: Inventory, desired_path: &Path) {
        let current = inventory.path_index.get(desired_path).copied().unwrap_or(0);
        self.inventory = inventory;
        self.current = current;
        self.selected = 0;
        self.scroll = 0;
        self.marked.clear();
        self.history.clear();
        self.inline_scan = None;
        self.confirm = None;
        self.clamp_selection();
    }

    fn replace_inventory_selecting_child(&mut self, inventory: Inventory, child_path: &Path) {
        self.inventory = inventory;
        self.current = 0;
        self.selected = 0;
        self.scroll = 0;
        self.marked.clear();
        self.history.clear();
        self.inline_scan = None;
        self.confirm = None;

        if let Some(child_id) = self.inventory.path_index.get(child_path).copied() {
            if let Some(position) = self.children().iter().position(|child| *child == child_id) {
                self.selected = self.parent_row_offset().saturating_add(position);
                self.history.push(ViewState {
                    current: child_id,
                    selected: 0,
                    scroll: 0,
                });
            }
        }

        self.clamp_selection();
    }

    fn handle_key(&mut self, key: KeyEvent) -> AppCommand {
        if is_ctrl_c(key) {
            self.should_quit = true;
            return AppCommand::None;
        }

        if self.confirm.is_some() {
            return self.handle_confirm_key(key);
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('v') => {
                    self.page_down();
                    return AppCommand::None;
                }
                KeyCode::Char('y') => {
                    self.page_up();
                    return AppCommand::None;
                }
                _ => {}
            }
        }

        match key.code {
            KeyCode::Esc => {
                self.undo_last_action();
                AppCommand::None
            }
            KeyCode::Char('r') => AppCommand::Rescan,
            KeyCode::Down => {
                self.move_selection(1, key.modifiers.contains(KeyModifiers::SHIFT));
                AppCommand::None
            }
            KeyCode::Up => {
                self.move_selection(-1, key.modifiers.contains(KeyModifiers::SHIFT));
                AppCommand::None
            }
            KeyCode::Char('j') => {
                self.move_selection(1, false);
                AppCommand::None
            }
            KeyCode::Char('k') => {
                self.move_selection(-1, false);
                AppCommand::None
            }
            KeyCode::PageDown => {
                self.page_down();
                AppCommand::None
            }
            KeyCode::PageUp => {
                self.page_up();
                AppCommand::None
            }
            KeyCode::Home => {
                self.selected = 0;
                self.keep_selection_visible();
                AppCommand::None
            }
            KeyCode::End => {
                let row_count = self.row_count();
                if row_count > 0 {
                    self.selected = row_count - 1;
                    self.keep_selection_visible();
                }
                AppCommand::None
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('o') => self.open_selected(),
            KeyCode::Left | KeyCode::Backspace => {
                self.go_parent();
                AppCommand::None
            }
            KeyCode::Char(' ') => {
                self.toggle_marked_selected();
                AppCommand::None
            }
            KeyCode::Char('x') => {
                if let Some(plan) = self.delete_plan() {
                    self.confirm = Some(DeleteConfirmation::new(plan));
                }
                AppCommand::None
            }
            _ => AppCommand::None,
        }
    }

    fn handle_inline_scan_key(&mut self, key: KeyEvent) -> AppCommand {
        if is_ctrl_c(key) {
            self.should_quit = true;
            return AppCommand::None;
        }

        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.move_selection(1, false);
                AppCommand::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_selection(-1, false);
                AppCommand::None
            }
            KeyCode::PageDown => {
                self.page_down();
                AppCommand::None
            }
            KeyCode::PageUp => {
                self.page_up();
                AppCommand::None
            }
            KeyCode::Home => {
                self.selected = 0;
                self.keep_selection_visible();
                AppCommand::None
            }
            KeyCode::End => {
                let row_count = self.row_count();
                if row_count > 0 {
                    self.selected = row_count - 1;
                    self.keep_selection_visible();
                }
                AppCommand::None
            }
            _ => AppCommand::None,
        }
    }

    fn handle_confirm_key(&mut self, key: KeyEvent) -> AppCommand {
        match key.code {
            KeyCode::Left | KeyCode::Char('h') => {
                if let Some(confirm) = &mut self.confirm {
                    confirm.choose_cancel();
                }
                AppCommand::None
            }
            KeyCode::Right | KeyCode::Char('l') => {
                if let Some(confirm) = &mut self.confirm {
                    confirm.choose_delete();
                }
                AppCommand::None
            }
            KeyCode::Tab | KeyCode::BackTab => {
                if let Some(confirm) = &mut self.confirm {
                    confirm.toggle();
                }
                AppCommand::None
            }
            KeyCode::Enter => self.activate_confirmation(),
            KeyCode::Char('y') if key.modifiers.is_empty() => self.confirm_delete(),
            KeyCode::Char('n') | KeyCode::Esc | KeyCode::Char('x') => {
                self.confirm = None;
                AppCommand::None
            }
            _ => AppCommand::None,
        }
    }

    fn activate_confirmation(&mut self) -> AppCommand {
        let Some(confirm) = self.confirm.take() else {
            return AppCommand::None;
        };

        match confirm.choice {
            ConfirmChoice::Cancel => AppCommand::None,
            ConfirmChoice::Delete => AppCommand::Delete(confirm.plan),
        }
    }

    fn confirm_delete(&mut self) -> AppCommand {
        if let Some(confirm) = self.confirm.take() {
            AppCommand::Delete(confirm.plan)
        } else {
            AppCommand::None
        }
    }

    fn handle_mouse(&mut self, kind: MouseEventKind) -> AppCommand {
        if self.confirm.is_some() {
            return AppCommand::None;
        }

        match kind {
            MouseEventKind::ScrollDown => {
                self.move_selection(3, false);
                AppCommand::None
            }
            MouseEventKind::ScrollUp => {
                self.move_selection(-3, false);
                AppCommand::None
            }
            _ => AppCommand::None,
        }
    }

    fn children(&self) -> &[usize] {
        &self.inventory.nodes[self.current].children
    }

    fn has_parent_row(&self) -> bool {
        if self.inventory.nodes[self.current].kind != NodeKind::Directory {
            return false;
        }

        self.inventory.nodes[self.current].parent.is_some()
            || self.inventory.nodes[self.current].path.parent().is_some()
    }

    fn parent_row_offset(&self) -> usize {
        usize::from(self.has_parent_row())
    }

    fn row_count(&self) -> usize {
        self.children()
            .len()
            .saturating_add(self.parent_row_offset())
    }

    fn selected_is_parent_row(&self) -> bool {
        self.has_parent_row() && self.selected == 0
    }

    fn selected_child_index(&self) -> Option<usize> {
        self.selected.checked_sub(self.parent_row_offset())
    }

    fn page_down(&mut self) {
        let delta = self.visible_rows.max(1) as isize;
        self.move_selection(delta, false);
    }

    fn page_up(&mut self) {
        let delta = self.visible_rows.max(1) as isize;
        self.move_selection(-delta, false);
    }

    fn move_selection(&mut self, delta: isize, mark: bool) {
        let row_count = self.row_count();
        if row_count == 0 {
            self.selected = 0;
            self.scroll = 0;
            return;
        }

        if mark {
            self.mark_selected();
        }

        let selected = self.selected as isize + delta;
        self.selected = selected.clamp(0, row_count.saturating_sub(1) as isize) as usize;
        if mark {
            self.mark_selected();
        }
        self.keep_selection_visible();
    }

    fn keep_selection_visible(&mut self) {
        let row_count = self.row_count();
        if row_count == 0 {
            self.selected = 0;
            self.scroll = 0;
            return;
        }

        let visible_rows = self.visible_rows.max(1);
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if self.selected >= self.scroll.saturating_add(visible_rows) {
            self.scroll = self.selected.saturating_sub(visible_rows - 1);
        }

        let max_scroll = row_count.saturating_sub(visible_rows);
        self.scroll = self.scroll.min(max_scroll);
    }

    fn clamp_selection(&mut self) {
        let row_count = self.row_count();
        if row_count == 0 {
            self.selected = 0;
            self.scroll = 0;
        } else {
            self.selected = self.selected.min(row_count - 1);
            self.keep_selection_visible();
        }
    }

    fn open_selected(&mut self) -> AppCommand {
        if self.selected_is_parent_row() {
            if self.inventory.nodes[self.current].parent.is_some() {
                self.go_parent();
                return AppCommand::None;
            }

            if let Some(parent_path) = self.inventory.root.parent().map(Path::to_path_buf) {
                return AppCommand::ScanParent(ParentScanRequest {
                    parent_path,
                    reuse: ReusableSubtree::from_inventory(&self.inventory),
                    select_path: self.inventory.root.clone(),
                });
            }

            return AppCommand::None;
        }

        let Some(node_id) = self.selected_node_id() else {
            return AppCommand::None;
        };
        if self.inventory.nodes[node_id].kind == NodeKind::Directory {
            self.push_history();
            self.current = node_id;
            self.selected = 0;
            self.scroll = 0;
            self.marked.clear();
        }
        AppCommand::None
    }

    fn go_parent(&mut self) {
        let Some(parent) = self.inventory.nodes[self.current].parent else {
            return;
        };
        self.push_history();
        let old_current = self.current;
        self.current = parent;
        self.marked.clear();
        self.selected = self
            .children()
            .iter()
            .position(|child| *child == old_current)
            .map(|position| self.parent_row_offset().saturating_add(position))
            .unwrap_or(0);
        self.keep_selection_visible();
    }

    fn toggle_marked_selected(&mut self) {
        let Some(node_id) = self.selected_node_id() else {
            return;
        };
        if !self.marked.insert(node_id) {
            self.marked.remove(&node_id);
        }
    }

    fn mark_selected(&mut self) {
        if let Some(node_id) = self.selected_node_id() {
            self.marked.insert(node_id);
        }
    }

    fn selected_node_id(&self) -> Option<usize> {
        let index = self.selected_child_index()?;
        self.children().get(index).copied()
    }

    fn selected_target(&self) -> Option<DeleteTarget> {
        let node_id = self.selected_node_id()?;
        self.delete_target_for_node(node_id)
    }

    fn delete_target_for_node(&self, node_id: usize) -> Option<DeleteTarget> {
        let node = &self.inventory.nodes[node_id];
        Some(DeleteTarget {
            path: node.path.to_path_buf(),
            display_name: node.name.clone(),
            total_size: node.total_size,
            kind: node.kind,
        })
    }

    fn marked_targets(&self) -> Vec<DeleteTarget> {
        self.children()
            .iter()
            .filter(|node_id| self.marked.contains(node_id))
            .filter_map(|node_id| self.delete_target_for_node(*node_id))
            .collect()
    }

    fn push_history(&mut self) {
        self.history.push(ViewState {
            current: self.current,
            selected: self.selected,
            scroll: self.scroll,
        });
    }

    fn undo_last_action(&mut self) {
        if !self.marked.is_empty() {
            self.marked.clear();
            return;
        }

        let Some(state) = self.history.pop() else {
            return;
        };
        self.current = state
            .current
            .min(self.inventory.nodes.len().saturating_sub(1));
        self.selected = state.selected;
        self.scroll = state.scroll;
        self.marked.clear();
        self.clamp_selection();
    }

    fn delete_plan(&self) -> Option<DeletePlan> {
        if self.marked.is_empty() {
            return self
                .selected_target()
                .and_then(|target| DeletePlan::from_targets(vec![target]));
        }

        DeletePlan::from_targets(self.marked_targets())
    }

    fn remove_deleted_targets(&mut self, deleted_paths: &[PathBuf]) -> usize {
        let mut target_ids = deleted_paths
            .iter()
            .filter_map(|path| self.inventory.path_index.get(path.as_path()).copied())
            .collect::<Vec<_>>();
        target_ids.sort_unstable_by_key(|node_id| path_depth(&self.inventory.nodes[*node_id].path));

        let mut removed_roots = 0usize;
        for node_id in target_ids {
            if self.remove_node_subtree(node_id) {
                removed_roots = removed_roots.saturating_add(1);
            }
        }

        self.marked = self
            .marked
            .iter()
            .copied()
            .filter(|node_id| self.node_is_live(*node_id))
            .collect();
        self.history = self
            .history
            .iter()
            .copied()
            .filter(|state| self.node_is_live(state.current))
            .collect();
        if !self.node_is_live(self.current) {
            self.current = 0;
            self.history.clear();
        }
        self.clamp_selection();

        removed_roots
    }

    fn remove_node_subtree(&mut self, node_id: usize) -> bool {
        if !self.node_is_live(node_id) {
            return false;
        }

        let removed_size = self.inventory.nodes[node_id].total_size;
        let removed_entries = self.inventory.nodes[node_id]
            .subtree_entries
            .saturating_add(1);
        let parent = self.inventory.nodes[node_id].parent;
        if let Some(parent_id) = parent {
            self.inventory.nodes[parent_id]
                .children
                .retain(|child| *child != node_id);
        }

        let mut ancestor = parent;
        while let Some(ancestor_id) = ancestor {
            self.inventory.nodes[ancestor_id].total_size = self.inventory.nodes[ancestor_id]
                .total_size
                .saturating_sub(removed_size);
            self.inventory.nodes[ancestor_id].subtree_entries = self.inventory.nodes[ancestor_id]
                .subtree_entries
                .saturating_sub(removed_entries);
            ancestor = self.inventory.nodes[ancestor_id].parent;
        }

        let subtree = self.subtree_node_ids(node_id);
        for removed_id in subtree {
            let removed_path = self.inventory.nodes[removed_id].path.clone();
            self.inventory.path_index.remove(&removed_path);
            self.inventory.nodes[removed_id].parent = None;
            self.inventory.nodes[removed_id].children.clear();
        }
        self.inventory.scanned_entries = self
            .inventory
            .scanned_entries
            .saturating_sub(removed_entries);

        true
    }

    fn subtree_node_ids(&self, root: usize) -> Vec<usize> {
        let mut subtree = Vec::new();
        let mut stack = vec![root];
        while let Some(node_id) = stack.pop() {
            subtree.push(node_id);
            stack.extend(self.inventory.nodes[node_id].children.iter().copied());
        }
        subtree
    }

    fn node_is_live(&self, node_id: usize) -> bool {
        self.inventory
            .nodes
            .get(node_id)
            .and_then(|node| self.inventory.path_index.get(node.path.as_ref()))
            .copied()
            == Some(node_id)
    }
}

struct Tui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl Tui {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("failed to enable raw mode")?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
            .context("failed to enter alternate screen")?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend).context("failed to create terminal")?;
        Ok(Self { terminal })
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        );
        let _ = self.terminal.show_cursor();
    }
}

fn draw(frame: &mut Frame<'_>, app: &mut App) {
    let area = frame.area();
    if app.inline_scan.is_some() {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Min(6),
                Constraint::Length(2),
            ])
            .split(area);

        draw_header(frame, chunks[0], app);
        draw_inline_scan(
            frame,
            chunks[1],
            app.inline_scan.as_ref().expect("inline scan exists"),
        );
        draw_table(frame, chunks[2], app);
        draw_footer(frame, chunks[3], app);
    } else {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(6),
                Constraint::Length(2),
            ])
            .split(area);

        draw_header(frame, chunks[0], app);
        draw_table(frame, chunks[1], app);
        draw_footer(frame, chunks[2], app);
    }

    if let Some(confirm) = &app.confirm {
        draw_confirm(frame, area, confirm);
    }
}

fn draw_scan_progress(
    frame: &mut Frame<'_>,
    path: &Path,
    progress: ScanProgress,
    started: Instant,
) {
    let area = frame.area();
    let elapsed = started.elapsed();
    let mut lines = vec![
        Line::from("Directory:"),
        Line::from(path.display().to_string()),
        Line::from(format!("Entries found: {}", progress.scanned_entries)),
    ];
    if progress.errors > 0 {
        lines.push(Line::from(format!("Errors: {}", progress.errors)));
    }
    lines.push(Line::from(format!("Elapsed: {}", format_duration(elapsed))));

    let panel = centered_fixed_rect(
        area.width.saturating_mul(4).saturating_div(5).max(40),
        lines.len().saturating_add(2) as u16,
        area,
    );
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true }),
        panel,
    );
}

fn draw_inline_scan(frame: &mut Frame<'_>, area: Rect, scan: &InlineScanStatus) {
    let elapsed = scan.started.elapsed();
    let spinner = match (elapsed.as_millis() / 120) % 4 {
        0 => "-",
        1 => "\\",
        2 => "|",
        _ => "/",
    };

    let mut spans = vec![
        Span::styled(
            format!("{spinner} Scanning parent"),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!("  {}", scan.path.display())),
        Span::raw(format!("  |  {} entries", scan.progress.scanned_entries)),
    ];
    if scan.progress.errors > 0 {
        spans.push(Span::raw(format!("  |  {} errors", scan.progress.errors)));
    }
    spans.push(Span::raw(format!(
        "  |  elapsed {}",
        format_duration(elapsed)
    )));

    let paragraph = Paragraph::new(Line::from(spans))
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
}

fn draw_header(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let current = &app.inventory.nodes[app.current];
    let title = Line::from(vec![
        Span::styled("Fast Disk Inventory", Style::default().fg(Color::Cyan)),
        Span::raw("  "),
        Span::styled(
            format_bytes(current.total_size),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
    ]);
    let mut detail_spans = vec![
        Span::raw(current.path.display().to_string()),
        Span::raw(format!(
            "  |  {} entries scanned in {}",
            app.inventory.scanned_entries,
            format_duration(app.inventory.elapsed)
        )),
    ];
    if !app.inventory.errors.is_empty() {
        detail_spans.push(Span::raw(format!(
            "  |  {} errors",
            app.inventory.errors.len()
        )));
    }
    let details = Line::from(detail_spans);

    let paragraph = Paragraph::new(vec![title, details])
        .block(Block::default().borders(Borders::BOTTOM))
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
}

fn draw_table(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let visible_rows = usize::from(area.height.saturating_sub(3)).max(1);
    app.visible_rows = visible_rows;
    app.keep_selection_visible();

    let parent_size = app.inventory.nodes[app.current].total_size;
    let row_count = app.row_count();
    let parent_row_offset = app.parent_row_offset();
    let end = row_count.min(app.scroll.saturating_add(visible_rows));

    let rows = (app.scroll..end).map(|row_index| {
        if app.has_parent_row() && row_index == 0 {
            return Row::new(vec![
                Cell::from(">"),
                Cell::from(""),
                Cell::from(NodeKind::Directory.label()),
                Cell::from(""),
                Cell::from(""),
                Cell::from(""),
                Cell::from(".."),
            ])
            .style(Style::default().bg(Color::Rgb(18, 36, 40)));
        }

        let child_index = row_index.saturating_sub(parent_row_offset);
        let node_id = app.children()[child_index];
        let node = &app.inventory.nodes[node_id];
        let is_directory = node.kind == NodeKind::Directory;
        let is_marked = app.marked.contains(&node_id);
        let percent = if parent_size > 0 {
            (node.total_size as f64 / parent_size as f64) * 100.0
        } else {
            0.0
        };
        let mut style = if is_directory {
            Style::default().bg(Color::Rgb(18, 36, 40))
        } else {
            Style::default()
        };
        if is_marked {
            style = style.fg(Color::Yellow).add_modifier(Modifier::BOLD);
        }

        Row::new(vec![
            Cell::from(if is_directory { ">" } else { "" }),
            Cell::from(if is_marked { "x" } else { "" }),
            Cell::from(node.kind.label()),
            Cell::from(format_bytes(node.total_size)),
            Cell::from(format!("{percent:>5.1}%")),
            Cell::from(node.subtree_entries.to_string()),
            Cell::from(node.name.clone()),
        ])
        .style(style)
    });

    let table = Table::new(
        rows,
        [
            Constraint::Length(2),
            Constraint::Length(3),
            Constraint::Length(6),
            Constraint::Length(12),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Min(24),
        ],
    )
    .header(
        Row::new(vec!["", "sel", "kind", "size", "share", "entries", "name"]).style(
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .block(Block::default().borders(Borders::ALL).title("Top Paths"))
    .row_highlight_style(
        Style::default()
            .fg(Color::Black)
            .bg(Color::White)
            .add_modifier(Modifier::BOLD),
    );

    let selected = if row_count == 0 {
        None
    } else {
        Some(app.selected.saturating_sub(app.scroll))
    };
    let mut state = TableState::default().with_selected(selected);
    frame.render_stateful_widget(table, area, &mut state);

    if row_count > visible_rows {
        let mut scrollbar_state = ScrollbarState::new(row_count).position(app.scroll);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("^"))
                .end_symbol(Some("v")),
            area,
            &mut scrollbar_state,
        );
    }
}

fn draw_footer(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let status = if app.status.is_empty() {
        let current = &app.inventory.nodes[app.current];
        format!(
            "{} direct items | {} total descendants | {} selected",
            current.children.len(),
            current.subtree_entries,
            app.marked.len()
        )
    } else {
        app.status.clone()
    };
    let paragraph = Paragraph::new(status)
        .block(Block::default().borders(Borders::TOP))
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
}

fn draw_confirm(frame: &mut Frame<'_>, area: Rect, confirm: &DeleteConfirmation) {
    let modal = centered_fixed_rect(72, confirm_modal_height(&confirm.plan), area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red))
        .title(Span::styled(
            " Confirm Delete ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ));
    frame.render_widget(block, modal);

    let inner = inset_rect(modal, 2, 1);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(3)])
        .split(inner);

    let text = if confirm.plan.targets.len() == 1 {
        confirm_text_for_one(&confirm.plan.targets[0], confirm.plan.total_size)
    } else {
        confirm_text_for_many(&confirm.plan)
    };

    let paragraph = Paragraph::new(text)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, chunks[0]);

    let buttons = Paragraph::new(confirm_button_line(confirm))
        .alignment(Alignment::Right)
        .wrap(Wrap { trim: true });
    frame.render_widget(buttons, chunks[1]);
}

fn confirm_text_for_one(target: &DeleteTarget, total_size: u64) -> Vec<Line<'static>> {
    let title = if target.kind == NodeKind::Directory {
        "Delete this directory?"
    } else {
        "Delete this file?"
    };
    let warning = if target.kind == NodeKind::Directory {
        "The directory and its contents will be removed."
    } else {
        "The selected file will be removed."
    };

    vec![
        Line::from(Span::styled(
            title,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            target.display_name.clone(),
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            target.path.display().to_string(),
            Style::default().fg(Color::Gray),
        )),
        Line::from(Span::styled(
            format!("Size: {}", format_bytes(total_size)),
            Style::default().fg(Color::Yellow),
        )),
        Line::from(""),
        Line::from(warning),
    ]
}

fn confirm_text_for_many(plan: &DeletePlan) -> Vec<Line<'static>> {
    let mut text = vec![
        Line::from(Span::styled(
            format!("Delete {} selected paths?", plan.targets.len()),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            format!("Total size: {}", format_bytes(plan.total_size)),
            Style::default().fg(Color::Yellow),
        )),
        Line::from(""),
        Line::from("This recursively deletes selected directories and files."),
        Line::from(""),
    ];

    for target in plan.targets.iter().take(4) {
        text.push(Line::from(format!("- {}", target.display_name)));
    }
    if plan.targets.len() > 4 {
        text.push(Line::from(format!(
            "...and {} more",
            plan.targets.len().saturating_sub(4)
        )));
    }

    text
}

fn confirm_modal_height(plan: &DeletePlan) -> u16 {
    if plan.targets.len() == 1 {
        return 14;
    }

    13u16
        .saturating_add(plan.targets.len().min(4) as u16)
        .saturating_add(u16::from(plan.targets.len() > 4))
}

fn confirm_button_line(confirm: &DeleteConfirmation) -> Line<'static> {
    Line::from(vec![
        confirm_button("Cancel", confirm.choice == ConfirmChoice::Cancel, false),
        Span::raw("  "),
        confirm_button("Yes", confirm.choice == ConfirmChoice::Delete, true),
    ])
}

fn confirm_button(label: &'static str, selected: bool, danger: bool) -> Span<'static> {
    let style = if selected && danger {
        Style::default()
            .fg(Color::White)
            .bg(Color::Red)
            .add_modifier(Modifier::BOLD)
    } else if selected {
        Style::default()
            .fg(Color::Black)
            .bg(Color::White)
            .add_modifier(Modifier::BOLD)
    } else if danger {
        Style::default()
            .fg(Color::LightRed)
            .bg(Color::Rgb(44, 24, 24))
    } else {
        Style::default().fg(Color::Gray).bg(Color::Rgb(32, 34, 38))
    };

    Span::styled(format!("  {label}  "), style)
}

fn centered_fixed_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

fn inset_rect(area: Rect, horizontal: u16, vertical: u16) -> Rect {
    Rect {
        x: area.x.saturating_add(horizontal),
        y: area.y.saturating_add(vertical),
        width: area.width.saturating_sub(horizontal.saturating_mul(2)),
        height: area.height.saturating_sub(vertical.saturating_mul(2)),
    }
}

fn is_ctrl_c(key: KeyEvent) -> bool {
    key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c')
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 7] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }

    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }

    if value >= 100.0 {
        format!("{value:.0} {}", UNITS[unit])
    } else if value >= 10.0 {
        format!("{value:.1} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs() >= 60 {
        let minutes = duration.as_secs() / 60;
        let seconds = duration.as_secs() % 60;
        format!("{minutes}m {seconds}s")
    } else if duration.as_secs() >= 1 {
        format!("{:.1}s", duration.as_secs_f64())
    } else {
        format!("{}ms", duration.as_millis())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static NEXT_TEMP_ID: AtomicUsize = AtomicUsize::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock should be after unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "disk-inventory-explorer-test-{}-{nanos}-{id}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("failed to create temp dir");
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn format_bytes_uses_binary_units() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1536), "1.50 KiB");
    }

    #[test]
    fn help_hides_advanced_scan_options_and_uses_lowercase_version_flag() {
        let help = cli_command().render_long_help().to_string();

        assert!(!help.contains("--one-file-system"));
        assert!(!help.contains("--threads"));
        assert!(!help.contains("-j,"));
        assert!(!help.contains("-V, --version"));
        assert!(help.contains("-v, --version"));
    }

    #[test]
    fn path_argument_does_not_require_version_flag() {
        let matches = cli_command()
            .try_get_matches_from(["die", "./"])
            .expect("path should parse without version flag");

        assert_eq!(
            matches.get_one::<PathBuf>("path"),
            Some(&PathBuf::from("./"))
        );
    }

    #[test]
    fn scan_rolls_up_sizes_and_sorts_children() {
        let temp = TempDir::new();
        let large_dir = temp.path.join("large");
        let small_dir = temp.path.join("small");
        fs::create_dir_all(&large_dir).expect("failed to create large dir");
        fs::create_dir_all(&small_dir).expect("failed to create small dir");
        fs::write(large_dir.join("large.bin"), vec![7u8; 128 * 1024])
            .expect("failed to write large file");
        fs::write(small_dir.join("small.bin"), [1u8; 16]).expect("failed to write small file");

        let inventory = scan_inventory(temp.path.clone(), 2, false).expect("scan failed");
        let root = &inventory.nodes[0];
        assert_eq!(root.children.len(), 2);

        let first = &inventory.nodes[root.children[0]];
        let second = &inventory.nodes[root.children[1]];
        assert_eq!(first.name, "large");
        assert_eq!(second.name, "small");
        assert!(first.total_size > second.total_size);
        assert!(root.total_size >= first.total_size + second.total_size);
    }

    #[test]
    fn fast_delete_removes_nested_directory() {
        let temp = TempDir::new();
        let target = temp.path.join("target");
        let nested = target.join("nested");
        fs::create_dir_all(&nested).expect("failed to create nested dir");
        fs::write(target.join("root.bin"), [1u8; 32]).expect("failed to write root file");
        fs::write(nested.join("child.bin"), [2u8; 32]).expect("failed to write child file");

        let stats = fast_delete(&target, 2, false).expect("delete failed");
        assert!(!target.exists());
        assert!(stats.errors.is_empty());
        assert_eq!(stats.removed_files, 2);
        assert!(stats.removed_dirs >= 2);
    }

    #[test]
    fn prune_nested_targets_keeps_parent_delete_only() {
        let parent = DeleteTarget {
            path: PathBuf::from("/tmp/fdi-parent"),
            display_name: "fdi-parent".to_string(),
            total_size: 8,
            kind: NodeKind::Directory,
        };
        let child = DeleteTarget {
            path: PathBuf::from("/tmp/fdi-parent/child"),
            display_name: "child".to_string(),
            total_size: 4,
            kind: NodeKind::File,
        };

        let targets = prune_nested_targets(vec![child, parent.clone()]);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].path, parent.path);
    }

    #[test]
    fn shift_move_marks_start_and_destination() {
        let root = PathBuf::from("/tmp/fdi-selection-root");
        let records = vec![
            EntryRecord {
                path: root.clone().into(),
                kind: NodeKind::Directory,
                own_size: 0,
            },
            EntryRecord {
                path: root.join("a").into(),
                kind: NodeKind::File,
                own_size: 1,
            },
            EntryRecord {
                path: root.join("b").into(),
                kind: NodeKind::File,
                own_size: 1,
            },
            EntryRecord {
                path: root.join("c").into(),
                kind: NodeKind::File,
                own_size: 1,
            },
        ];
        let inventory = build_inventory(root, records);
        let mut app = App::new(inventory, 2, false);
        select_first_child(&mut app);

        app.move_selection(1, true);

        assert_eq!(app.selected, app.parent_row_offset() + 1);
        assert_eq!(app.marked.len(), 2);
    }

    #[test]
    fn parent_row_is_first_row_when_directory_has_parent() {
        let app = selection_app();

        assert!(app.has_parent_row());
        assert!(app.selected_is_parent_row());
        assert_eq!(app.row_count(), app.children().len() + 1);
        assert!(app.selected_node_id().is_none());
    }

    #[test]
    fn enter_on_parent_row_inside_inventory_moves_to_existing_parent() {
        let mut app = navigation_app();
        select_first_child(&mut app);
        app.handle_key(key(KeyCode::Enter));
        assert_ne!(app.current, 0);

        let command = app.handle_key(key(KeyCode::Enter));

        assert!(matches!(command, AppCommand::None));
        assert_eq!(app.current, 0);
        assert_eq!(app.selected, app.parent_row_offset());
    }

    #[test]
    fn enter_on_parent_row_at_scan_root_requests_parent_scan() {
        let mut app = selection_app();
        let root = app.inventory.root.clone();

        let command = app.handle_key(key(KeyCode::Enter));

        let AppCommand::ScanParent(request) = command else {
            panic!("expected parent scan command");
        };
        assert_eq!(request.parent_path, root.parent().unwrap());
        assert_eq!(request.select_path, root);
        assert_eq!(request.reuse.root, request.select_path);
    }

    #[test]
    fn inline_scan_mode_keeps_explorer_responsive_without_opening_rows() {
        let mut app = selection_app();
        app.inline_scan = Some(InlineScanStatus {
            path: app.inventory.root.parent().unwrap().to_path_buf(),
            started: Instant::now(),
            progress: ScanProgress::default(),
        });

        let command = app.handle_inline_scan_key(key(KeyCode::Enter));
        assert!(matches!(command, AppCommand::None));
        assert!(app.selected_is_parent_row());

        app.handle_inline_scan_key(key(KeyCode::Down));
        assert_eq!(app.selected, 1);

        app.handle_inline_scan_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(app.should_quit);
    }

    #[test]
    fn parent_scan_reuses_existing_subtree_records() {
        let temp = TempDir::new();
        let child = temp.path.join("child");
        let nested = child.join("nested.txt");
        let sibling = temp.path.join("sibling.txt");
        fs::create_dir_all(&child).expect("failed to create child dir");
        fs::write(&nested, [1u8; 64]).expect("failed to write nested file");

        let child_inventory = scan_inventory(child.clone(), 2, false).expect("child scan failed");
        let reused_nested = child_inventory.root.join("nested.txt");
        fs::remove_file(&nested).expect("failed to remove nested file after scan");
        fs::write(&sibling, [2u8; 16]).expect("failed to write sibling file");
        let scanned_sibling = fs::canonicalize(&sibling).expect("failed to canonicalize sibling");

        let parent_inventory = scan_inventory_with_reuse(
            temp.path.clone(),
            2,
            false,
            Some(ReusableSubtree::from_inventory(&child_inventory)),
            None,
        )
        .expect("parent scan failed");

        assert!(
            parent_inventory
                .path_index
                .contains_key(reused_nested.as_path())
        );
        assert!(
            parent_inventory
                .path_index
                .contains_key(scanned_sibling.as_path())
        );
    }

    #[test]
    fn q_and_escape_do_not_quit_explorer() {
        let mut app = selection_app();

        app.handle_key(key(KeyCode::Char('q')));
        assert!(!app.should_quit);

        app.handle_key(key(KeyCode::Esc));
        assert!(!app.should_quit);
    }

    #[test]
    fn ctrl_c_quits_explorer() {
        let mut app = selection_app();

        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));

        assert!(app.should_quit);
    }

    #[test]
    fn delete_confirmation_enter_defaults_to_cancel() {
        let mut app = selection_app();
        select_first_child(&mut app);

        app.handle_key(key(KeyCode::Char('x')));
        assert_eq!(
            app.confirm.as_ref().map(|confirm| confirm.choice),
            Some(ConfirmChoice::Cancel)
        );

        let command = app.handle_key(key(KeyCode::Enter));

        assert!(matches!(command, AppCommand::None));
        assert!(app.confirm.is_none());
    }

    #[test]
    fn delete_confirmation_can_select_yes_with_arrow_and_enter() {
        let mut app = selection_app();
        select_first_child(&mut app);

        app.handle_key(key(KeyCode::Char('x')));
        app.handle_key(key(KeyCode::Right));
        let command = app.handle_key(key(KeyCode::Enter));

        let AppCommand::Delete(plan) = command else {
            panic!("expected delete command");
        };
        assert_eq!(plan.targets.len(), 1);
        assert_eq!(plan.targets[0].display_name, "a");
    }

    #[test]
    fn removing_deleted_target_updates_inventory_without_rescan() {
        let root = PathBuf::from("/tmp/fdi-delete-update-root");
        let deleted_dir = root.join("a");
        let kept_file = root.join("b");
        let records = vec![
            EntryRecord {
                path: root.clone().into(),
                kind: NodeKind::Directory,
                own_size: 10,
            },
            EntryRecord {
                path: deleted_dir.clone().into(),
                kind: NodeKind::Directory,
                own_size: 2,
            },
            EntryRecord {
                path: deleted_dir.join("child").into(),
                kind: NodeKind::File,
                own_size: 8,
            },
            EntryRecord {
                path: kept_file.clone().into(),
                kind: NodeKind::File,
                own_size: 4,
            },
        ];
        let mut app = App::new(build_inventory(root.clone(), records), 2, false);
        app.inventory.scanned_entries = app.inventory.nodes.len();

        let removed = app.remove_deleted_targets(&[deleted_dir.clone()]);

        assert_eq!(removed, 1);
        assert!(!app.inventory.path_index.contains_key(deleted_dir.as_path()));
        assert!(
            !app.inventory
                .path_index
                .contains_key(deleted_dir.join("child").as_path())
        );
        assert!(app.inventory.path_index.contains_key(kept_file.as_path()));
        assert_eq!(app.inventory.nodes[0].children.len(), 1);
        assert_eq!(app.inventory.nodes[0].total_size, 14);
        assert_eq!(app.inventory.nodes[0].subtree_entries, 1);
        assert_eq!(app.inventory.scanned_entries, 2);
    }

    #[test]
    fn escape_clears_marked_paths() {
        let mut app = selection_app();
        select_first_child(&mut app);
        app.toggle_marked_selected();
        assert_eq!(app.marked.len(), 1);

        app.handle_key(key(KeyCode::Esc));

        assert!(app.marked.is_empty());
        assert_eq!(app.current, 0);
        assert!(!app.should_quit);
    }

    #[test]
    fn escape_restores_previous_directory_view() {
        let mut app = navigation_app();
        select_first_child(&mut app);

        app.handle_key(key(KeyCode::Enter));
        assert_ne!(app.current, 0);

        app.handle_key(key(KeyCode::Esc));

        assert_eq!(app.current, 0);
        assert_eq!(app.selected, app.parent_row_offset());
        assert!(!app.should_quit);
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn select_first_child(app: &mut App) {
        app.selected = app.parent_row_offset();
    }

    fn selection_app() -> App {
        let root = PathBuf::from("/tmp/fdi-selection-root");
        let records = vec![
            EntryRecord {
                path: root.clone().into(),
                kind: NodeKind::Directory,
                own_size: 0,
            },
            EntryRecord {
                path: root.join("a").into(),
                kind: NodeKind::File,
                own_size: 1,
            },
            EntryRecord {
                path: root.join("b").into(),
                kind: NodeKind::File,
                own_size: 1,
            },
        ];
        App::new(build_inventory(root, records), 2, false)
    }

    fn navigation_app() -> App {
        let root = PathBuf::from("/tmp/fdi-navigation-root");
        let child = root.join("child");
        let records = vec![
            EntryRecord {
                path: root.clone().into(),
                kind: NodeKind::Directory,
                own_size: 0,
            },
            EntryRecord {
                path: child.clone().into(),
                kind: NodeKind::Directory,
                own_size: 0,
            },
            EntryRecord {
                path: child.join("file").into(),
                kind: NodeKind::File,
                own_size: 1,
            },
        ];
        App::new(build_inventory(root, records), 2, false)
    }
}
