//! A dependency-free view of a grid search.
//!
//! [`GridView`] accumulates a search's [`RecordedEvent`](crate::RecordedEvent)
//! stream into the sets a viewer needs (expanded, frontier, current, path) plus
//! the best `g` cost seen at each cell. This is the shared core that backends
//! build on: the terminal TUI colors the same [`CellKind`]s (and can show
//! [`GridView::cost_at`] in a detailed mode), and [`GridView::render_ascii`] gives
//! a deterministic string for snapshot tests — one data model, many backends.

use std::collections::{HashMap, HashSet};

use crate::grid::{GridCost, GridState};
use crate::observe::RecordedEvent;

/// What a cell represents in the current view, in priority order. The same kind
/// drives both the ASCII glyph and the TUI color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellKind {
    /// An obstacle.
    Blocked,
    /// The start cell.
    Start,
    /// The goal cell.
    Goal,
    /// The node currently being expanded.
    Current,
    /// A cell on the final path.
    Path,
    /// An already-expanded (closed) node.
    Expanded,
    /// A discovered but not-yet-expanded (open) node.
    Frontier,
    /// Free, untouched space.
    Empty,
}

impl CellKind {
    /// The single character used to draw this kind in ASCII.
    pub fn glyph(self) -> char {
        match self {
            CellKind::Blocked => '#',
            CellKind::Start => 'S',
            CellKind::Goal => 'G',
            CellKind::Current => '@',
            CellKind::Path => '*',
            CellKind::Expanded => '.',
            CellKind::Frontier => '+',
            CellKind::Empty => '·',
        }
    }
}

/// Accumulates a grid search's progress for rendering.
#[derive(Debug, Clone)]
pub struct GridView {
    width: i32,
    height: i32,
    blocked: HashSet<GridState>,
    start: GridState,
    goal: GridState,
    expanded: HashSet<GridState>,
    frontier: HashSet<GridState>,
    current: Option<GridState>,
    path: Vec<GridState>,
    g_of: HashMap<GridState, u32>,
}

impl GridView {
    /// A fresh view of a `width × height` grid with the given obstacles.
    pub fn new(
        width: i32,
        height: i32,
        blocked: impl IntoIterator<Item = GridState>,
        start: GridState,
        goal: GridState,
    ) -> Self {
        Self {
            width,
            height,
            blocked: blocked.into_iter().collect(),
            start,
            goal,
            expanded: HashSet::new(),
            frontier: HashSet::new(),
            current: None,
            path: Vec::new(),
            g_of: HashMap::new(),
        }
    }

    /// Grid width in cells.
    pub fn width(&self) -> i32 {
        self.width
    }

    /// Grid height in cells.
    pub fn height(&self) -> i32 {
        self.height
    }

    /// Advance the view by one recorded event.
    pub fn apply(&mut self, event: &RecordedEvent<GridState, GridCost>) {
        match event {
            RecordedEvent::Expanded { state, g, .. } => {
                self.current = Some(*state);
                self.frontier.remove(state);
                self.expanded.insert(*state);
                self.g_of.insert(*state, g.get());
            }
            RecordedEvent::Generated { state, g, .. }
            | RecordedEvent::Improved { state, g, .. } => {
                self.g_of.insert(*state, g.get());
                if !self.expanded.contains(state) {
                    self.frontier.insert(*state);
                }
            }
            _ => {}
        }
    }

    /// Reveal a final path (e.g. once replay finishes) and clear the cursor.
    pub fn reveal_path(&mut self, path: &[GridState]) {
        self.path = path.to_vec();
        self.current = None;
    }

    /// The best `g` cost seen at a cell, if the search has reached it. (Raw
    /// fixed-point value — `GridCost::get`.)
    pub fn cost_at(&self, cell: GridState) -> Option<u32> {
        self.g_of.get(&cell).copied()
    }

    /// Classify a cell for rendering.
    pub fn cell_kind(&self, cell: GridState) -> CellKind {
        if self.blocked.contains(&cell) {
            CellKind::Blocked
        } else if cell == self.start {
            CellKind::Start
        } else if cell == self.goal {
            CellKind::Goal
        } else if self.current == Some(cell) {
            CellKind::Current
        } else if self.path.contains(&cell) {
            CellKind::Path
        } else if self.expanded.contains(&cell) {
            CellKind::Expanded
        } else if self.frontier.contains(&cell) {
            CellKind::Frontier
        } else {
            CellKind::Empty
        }
    }

    /// Render the whole grid as text, one row per line, `y` increasing upward.
    /// Deterministic — suitable as a snapshot. (The compact TUI view.)
    pub fn render_ascii(&self) -> String {
        let mut out = String::new();
        for y in (0..self.height).rev() {
            for x in 0..self.width {
                if x > 0 {
                    out.push(' ');
                }
                out.push(self.cell_kind(GridState::new(x, y)).glyph());
            }
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(x: i32, y: i32) -> GridState {
        GridState::new(x, y)
    }

    #[test]
    fn renders_initial_grid_with_landmarks() {
        let view = GridView::new(3, 3, [g(1, 1)], g(0, 0), g(2, 2));
        let expected = "\
· · G
· # ·
S · ·
";
        assert_eq!(view.render_ascii(), expected);
    }

    #[test]
    fn apply_moves_a_cell_from_frontier_to_current_and_records_cost() {
        let mut view = GridView::new(3, 1, [], g(0, 0), g(2, 0));

        view.apply(&RecordedEvent::Generated {
            state: g(1, 0),
            g: GridCost::raw(100),
            f: GridCost::raw(200),
        });
        assert_eq!(view.cell_kind(g(1, 0)), CellKind::Frontier);
        assert_eq!(view.cost_at(g(1, 0)), Some(100));

        view.apply(&RecordedEvent::Expanded {
            state: g(1, 0),
            g: GridCost::raw(100),
            f: GridCost::raw(200),
        });
        // Current takes priority over Expanded for the just-popped node.
        assert_eq!(view.cell_kind(g(1, 0)), CellKind::Current);
    }

    #[test]
    fn revealed_path_is_drawn() {
        let mut view = GridView::new(3, 1, [], g(0, 0), g(2, 0));
        view.reveal_path(&[g(0, 0), g(1, 0), g(2, 0)]);
        assert_eq!(view.cell_kind(g(1, 0)), CellKind::Path);
    }
}
