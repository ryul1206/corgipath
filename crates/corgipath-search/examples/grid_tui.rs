//! Getting-started TUI: watch A\* solve a grid, step by step, in your terminal —
//! no Rerun, no setup. You **assemble** a scenario (pick the connectivity), then
//! visualize that finished assembly. Run with:
//!
//! ```text
//! cargo run -p corgipath-search --example grid_tui            # 8-connected (diagonals)
//! cargo run -p corgipath-search --example grid_tui -- --four  # 4-connected
//! ```
//!
//! Controls: `space` play/pause · `→` single step · `r` restart · `q`/`Esc` quit.
//!
//! How it works: we assemble the space + the matching heuristic (via
//! [`grid_heuristic`], so the pairing can't be wrong), run the search once with a
//! [`Recorder`], then *replay* the recorded trace stream into a [`GridView`] —
//! the algorithm itself knows nothing about the display.

use std::time::Duration;

use corgipath_core::{
    CellKind, Connectivity, GridCollision, GridCost, GridSpace, GridState, GridView, Outcome,
    RecordedEvent, Recorder,
};
use corgipath_search::{astar, grid_heuristic, Unbounded};

use ratatui::crossterm::event::{self, Event, KeyCode};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use ratatui::{DefaultTerminal, Frame};

fn at(x: i32, y: i32) -> GridState {
    GridState::new(x, y)
}

/// A demo scenario: two walls, each with a single gap, forcing a detour.
fn scenario() -> (i32, i32, Vec<GridState>, GridState, GridState) {
    let (w, h) = (12, 8);
    let mut blocked = Vec::new();
    // Vertical wall at x = 4, gap at y = 2.
    for y in 0..h {
        if y != 2 {
            blocked.push(at(4, y));
        }
    }
    // Vertical wall at x = 8, gap at y = 6.
    for y in 0..h {
        if y != 6 {
            blocked.push(at(8, y));
        }
    }
    (w, h, blocked, at(0, 0), at(11, 7))
}

/// How the grid is drawn. Cycled live with `v` — a display-only choice, separate
/// from the assembled scenario.
#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    /// One character per cell.
    Compact,
    /// Borderless color blocks (obstacles filled); cell gaps left for future use.
    Expanded,
    /// Expanded, plus the `g` cost inside reached cells.
    Detailed,
}

impl ViewMode {
    fn next(self) -> Self {
        match self {
            ViewMode::Compact => ViewMode::Expanded,
            ViewMode::Expanded => ViewMode::Detailed,
            ViewMode::Detailed => ViewMode::Compact,
        }
    }

    fn label(self) -> &'static str {
        match self {
            ViewMode::Compact => "compact",
            ViewMode::Expanded => "expanded",
            ViewMode::Detailed => "detailed",
        }
    }
}

struct App {
    mode: ViewMode,
    width: i32,
    height: i32,
    blocked: Vec<GridState>,
    start: GridState,
    goal: GridState,
    events: Vec<RecordedEvent<GridState, GridCost>>,
    path: Vec<GridState>,
    outcome: Outcome,
    view: GridView,
    idx: usize,
    paused: bool,
}

impl App {
    fn new(connectivity: Connectivity) -> Self {
        let (width, height, blocked, start, goal) = scenario();
        // Assemble the finished scenario: space + the heuristic that matches its
        // connectivity (so the pairing can never be inadmissible).
        let space = GridSpace::new(width, height).with_connectivity(connectivity);
        let col = GridCollision::with_blocked(blocked.clone());

        let mut rec = Recorder::new();
        let result = astar(
            &space,
            &col,
            start,
            goal,
            &grid_heuristic(&space, goal),
            &mut rec,
            &mut Unbounded,
        );

        let view = GridView::new(width, height, blocked.clone(), start, goal);
        Self {
            mode: ViewMode::Compact,
            width,
            height,
            blocked,
            start,
            goal,
            events: rec.into_events(),
            path: result.path.map(|p| p.states).unwrap_or_default(),
            outcome: result.outcome,
            view,
            idx: 0,
            paused: false,
        }
    }

    fn restart(&mut self) {
        self.view = GridView::new(
            self.width,
            self.height,
            self.blocked.clone(),
            self.start,
            self.goal,
        );
        self.idx = 0;
    }

    fn at_end(&self) -> bool {
        self.idx >= self.events.len()
    }

    fn step(&mut self) {
        if self.idx < self.events.len() {
            self.view.apply(&self.events[self.idx]);
            self.idx += 1;
            if self.at_end() {
                self.view.reveal_path(&self.path);
            }
        }
    }
}

fn style_for(kind: CellKind) -> Style {
    match kind {
        CellKind::Blocked => Style::new().fg(Color::DarkGray),
        CellKind::Start => Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        CellKind::Goal => Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        CellKind::Current => Style::new()
            .fg(Color::Black)
            .bg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
        CellKind::Path => Style::new().fg(Color::Green).add_modifier(Modifier::BOLD),
        CellKind::Expanded => Style::new().fg(Color::Blue),
        CellKind::Frontier => Style::new().fg(Color::LightBlue),
        CellKind::Empty => Style::new().fg(Color::DarkGray),
    }
}

/// Glyph + filled-block style for the expanded/detailed views. Non-obstacle tiles
/// have no border — just color; obstacles are a solid block. The gaps between
/// blocks are intentionally left empty (a future home for facing arrows or the
/// search-spread direction).
fn block_cell(kind: CellKind) -> (char, Style) {
    let s = Style::new();
    match kind {
        CellKind::Empty => (' ', s),
        CellKind::Blocked => (' ', s.bg(Color::DarkGray)),
        CellKind::Start => (
            'S',
            s.bg(Color::Cyan)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        ),
        CellKind::Goal => (
            'G',
            s.bg(Color::Magenta)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        ),
        CellKind::Current => (
            '@',
            s.bg(Color::Yellow)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        ),
        CellKind::Path => (' ', s.bg(Color::Green).fg(Color::Black)),
        CellKind::Expanded => (' ', s.bg(Color::Blue).fg(Color::White)),
        CellKind::Frontier => (' ', s.bg(Color::LightBlue).fg(Color::Black)),
    }
}

/// The compact view: one character per cell.
fn compact_lines(view: &GridView) -> Vec<Line<'static>> {
    let mut lines = Vec::with_capacity(view.height() as usize);
    for y in (0..view.height()).rev() {
        let mut spans = Vec::with_capacity(view.width() as usize);
        for x in 0..view.width() {
            let kind = view.cell_kind(at(x, y));
            spans.push(Span::styled(format!("{} ", kind.glyph()), style_for(kind)));
        }
        lines.push(Line::from(spans));
    }
    lines
}

/// The expanded view: borderless color blocks, with the `g` cost inside reached
/// cells when `detailed`.
fn block_lines(view: &GridView, detailed: bool) -> Vec<Line<'static>> {
    let cell_w = if detailed { 5 } else { 3 };
    let mut lines = Vec::new();
    for y in (0..view.height()).rev() {
        let mut spans = Vec::new();
        for x in 0..view.width() {
            let cell = at(x, y);
            let kind = view.cell_kind(cell);
            let (glyph, style) = block_cell(kind);
            let label = if detailed
                && matches!(
                    kind,
                    CellKind::Expanded | CellKind::Frontier | CellKind::Path
                ) {
                view.cost_at(cell)
                    .map(|g| g.to_string())
                    .unwrap_or_else(|| glyph.to_string())
            } else {
                glyph.to_string()
            };
            spans.push(Span::styled(format!("{label:^cell_w$}"), style));
            spans.push(Span::raw(" ")); // gap = the "border region", reserved for later
        }
        lines.push(Line::from(spans));
        lines.push(Line::from("")); // vertical gap
    }
    lines
}

fn ui(frame: &mut Frame, app: &App) {
    let chunks = Layout::vertical([Constraint::Min(1), Constraint::Length(3)]).split(frame.area());

    let lines = match app.mode {
        ViewMode::Compact => compact_lines(&app.view),
        ViewMode::Expanded => block_lines(&app.view, false),
        ViewMode::Detailed => block_lines(&app.view, true),
    };
    let grid = Paragraph::new(lines).block(Block::bordered().title(" corgipath · A* on a grid "));
    frame.render_widget(grid, chunks[0]);

    let status = format!(
        " {}/{}  {:?}  [space]{}  [->]step  [v]view:{}  [r]restart  [q]quit ",
        app.idx,
        app.events.len(),
        app.outcome,
        if app.paused { "play" } else { "pause" },
        app.mode.label(),
    );
    frame.render_widget(Paragraph::new(status).block(Block::bordered()), chunks[1]);
}

fn run(terminal: &mut DefaultTerminal, app: &mut App) -> std::io::Result<()> {
    loop {
        terminal.draw(|frame| ui(frame, app))?;

        // Poll with a timeout so the replay auto-advances when not paused.
        if event::poll(Duration::from_millis(120))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char(' ') => app.paused = !app.paused,
                    KeyCode::Char('v') => app.mode = app.mode.next(),
                    KeyCode::Right => app.step(),
                    KeyCode::Char('r') => app.restart(),
                    _ => {}
                }
            }
        } else if !app.paused && !app.at_end() {
            app.step();
        }
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    // Assemble-time choice: 8-connected by default, `--four` for 4-connected.
    let four = std::env::args().any(|a| a == "--four" || a == "4");
    let connectivity = if four {
        Connectivity::Four
    } else {
        Connectivity::Eight
    };

    let mut terminal = ratatui::init();
    let mut app = App::new(connectivity);
    let result = run(&mut terminal, &mut app);
    ratatui::restore();
    result
}
