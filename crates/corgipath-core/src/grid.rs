//! A minimal reference implementation of the core ports: a grid ([`GridSpace`],
//! 4- or 8-connected) and a blocked-cell collision checker ([`GridCollision`]).
//!
//! Batteries-included so the search algorithms have something concrete to run
//! against, and a worked example of how to implement [`SearchSpace`] /
//! [`Collision`]. The motion model (which neighbors exist) lives in the space;
//! the obstacles live in the collision checker — kept orthogonal on purpose.
//!
//! Costs use [`FixedPoint`]`<100>` (`GridCost`): one orthogonal step is `100`
//! and one diagonal step is `141` (≈ √2), so 8-connected paths are measured with
//! the correct octile geometry while staying in clean integers.

use crate::fixed::FixedPoint;
use crate::space::{Collision, SearchSpace};
use std::collections::HashSet;

/// The grid's cost type: fixed-point with `1.0 == 100`.
pub type GridCost = FixedPoint<100>;

/// Cost of one orthogonal (N/S/E/W) step: `1.0`.
pub const CARDINAL: GridCost = FixedPoint(100);
/// Cost of one diagonal step: `√2 ≈ 1.41`.
pub const DIAGONAL: GridCost = FixedPoint(141);

/// How a grid cell connects to its neighbors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// 4 neighbors: N, S, E, W.
    Four,
    /// 8 neighbors: the four orthogonals plus the four diagonals.
    Eight,
}

/// A cell on an integer grid. Discrete, so it is its own identity (`Id = State`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GridState {
    pub x: i32,
    pub y: i32,
}

impl GridState {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// A `width × height` grid, 4- or 8-connected, with fixed-point step costs.
///
/// Motion model only: it yields the in-bounds neighbors of a cell. Whether a cell
/// is *blocked* is a separate concern — see [`GridCollision`].
#[derive(Debug, Clone)]
pub struct GridSpace {
    width: i32,
    height: i32,
    connectivity: Connectivity,
}

impl GridSpace {
    /// Create a `width × height` grid, 4-connected by default. Both dimensions
    /// must be positive.
    pub fn new(width: i32, height: i32) -> Self {
        assert!(
            width > 0 && height > 0,
            "grid dimensions must be positive, got {width}x{height}"
        );
        Self {
            width,
            height,
            connectivity: Connectivity::Four,
        }
    }

    /// Set the connectivity (builder style).
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// This grid's connectivity. The matching admissible heuristic is chosen from
    /// it (see `corgipath_search::grid_heuristic`).
    pub fn connectivity(&self) -> Connectivity {
        self.connectivity
    }

    /// Width in cells.
    pub fn width(&self) -> i32 {
        self.width
    }

    /// Height in cells.
    pub fn height(&self) -> i32 {
        self.height
    }

    /// Whether a cell lies inside the grid.
    pub fn in_bounds(&self, cell: &GridState) -> bool {
        (0..self.width).contains(&cell.x) && (0..self.height).contains(&cell.y)
    }
}

impl SearchSpace for GridSpace {
    type State = GridState;
    type Id = GridState; // discrete: a cell is its own identity
    type Cost = GridCost;

    fn id(&self, state: &Self::State) -> Self::Id {
        *state
    }

    fn successors(&self, state: &Self::State, out: &mut Vec<(Self::State, Self::Cost)>) {
        out.clear();
        const CARDINAL_STEPS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        for (dx, dy) in CARDINAL_STEPS {
            let next = GridState::new(state.x + dx, state.y + dy);
            if self.in_bounds(&next) {
                out.push((next, CARDINAL));
            }
        }
        // One predictable branch per expansion (connectivity is constant during a
        // search); the cost is effectively free after branch prediction.
        if self.connectivity == Connectivity::Eight {
            const DIAGONAL_STEPS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
            for (dx, dy) in DIAGONAL_STEPS {
                let next = GridState::new(state.x + dx, state.y + dy);
                if self.in_bounds(&next) {
                    out.push((next, DIAGONAL));
                }
            }
        }
    }
}

/// Blocked-cell collision for a grid.
///
/// A single-cell footprint for now: a cell is in collision iff it was marked
/// blocked. A robot whose body spans several cells would, in a richer checker,
/// test every cell its footprint covers here — the [`Collision`] contract is
/// unchanged either way.
#[derive(Debug, Default, Clone)]
pub struct GridCollision {
    blocked: HashSet<GridState>,
}

impl GridCollision {
    /// An empty checker — nothing is blocked.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a checker from a set of blocked cells.
    pub fn with_blocked(cells: impl IntoIterator<Item = GridState>) -> Self {
        Self {
            blocked: cells.into_iter().collect(),
        }
    }

    /// Mark a cell blocked.
    pub fn block(&mut self, cell: GridState) {
        self.blocked.insert(cell);
    }

    /// How many cells are blocked.
    pub fn blocked_count(&self) -> usize {
        self.blocked.len()
    }
}

impl Collision<GridState> for GridCollision {
    fn is_colliding(&self, state: &GridState) -> bool {
        self.blocked.contains(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(out: &[(GridState, GridCost)]) -> Vec<GridState> {
        out.iter().map(|(s, _)| *s).collect()
    }

    #[test]
    fn successors_in_the_middle_are_the_four_neighbors() {
        let space = GridSpace::new(10, 10);
        let mut out = Vec::new();
        space.successors(&GridState::new(5, 5), &mut out);

        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|(_, c)| *c == CARDINAL));
        let got = ids(&out);
        for expected in [
            GridState::new(6, 5),
            GridState::new(4, 5),
            GridState::new(5, 6),
            GridState::new(5, 4),
        ] {
            assert!(got.contains(&expected), "missing {expected:?}");
        }
    }

    #[test]
    fn eight_connected_adds_four_diagonals() {
        let space = GridSpace::new(10, 10).with_connectivity(Connectivity::Eight);
        let mut out = Vec::new();
        space.successors(&GridState::new(5, 5), &mut out);

        assert_eq!(out.len(), 8);
        let cardinal = out.iter().filter(|(_, c)| *c == CARDINAL).count();
        let diagonal = out.iter().filter(|(_, c)| *c == DIAGONAL).count();
        assert_eq!(cardinal, 4);
        assert_eq!(diagonal, 4);
        assert!(ids(&out).contains(&GridState::new(6, 6)));
    }

    #[test]
    fn successors_at_a_corner_are_clipped_to_bounds() {
        let space = GridSpace::new(10, 10);
        let mut out = Vec::new();
        space.successors(&GridState::new(0, 0), &mut out);

        // Only (1,0) and (0,1) are in bounds; (-1,0) and (0,-1) are off-grid.
        assert_eq!(out.len(), 2);
        let got = ids(&out);
        assert!(got.contains(&GridState::new(1, 0)));
        assert!(got.contains(&GridState::new(0, 1)));
    }

    #[test]
    fn successors_clears_previous_contents() {
        let space = GridSpace::new(10, 10);
        let mut out = vec![(GridState::new(99, 99), FixedPoint(7))]; // garbage
        space.successors(&GridState::new(0, 0), &mut out);
        assert!(!ids(&out).contains(&GridState::new(99, 99)));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn buffer_is_reused_without_reallocating() {
        // A reused buffer's capacity converges and then stops growing.
        let space = GridSpace::new(10, 10);
        let mut out = Vec::new();

        space.successors(&GridState::new(5, 5), &mut out);
        assert_eq!(out.len(), 4);
        let cap = out.capacity();

        space.successors(&GridState::new(4, 4), &mut out);
        assert_eq!(out.len(), 4);
        assert_eq!(out.capacity(), cap, "reused buffer must not reallocate");
    }

    #[test]
    fn id_of_a_discrete_cell_is_itself() {
        let space = GridSpace::new(3, 3);
        let cell = GridState::new(2, 1);
        assert_eq!(space.id(&cell), cell);
    }

    #[test]
    fn grid_collision_reports_blocked_cells() {
        let blocked = GridState::new(2, 2);
        let col = GridCollision::with_blocked([blocked]);
        assert!(col.is_colliding(&blocked));
        assert!(!col.is_colliding(&GridState::new(0, 0)));
        assert_eq!(col.blocked_count(), 1);
    }

    // Proves the two orthogonal ports compose under generic bounds — the shape the
    // search algorithm will use: expand, then filter by collision.
    fn free_successors<Sp, C>(space: &Sp, col: &C, state: &Sp::State) -> Vec<Sp::State>
    where
        Sp: SearchSpace,
        C: Collision<Sp::State>,
    {
        let mut buf = Vec::new();
        space.successors(state, &mut buf);
        buf.into_iter()
            .filter(|(s, _)| !col.is_colliding(s))
            .map(|(s, _)| s)
            .collect()
    }

    #[test]
    fn space_and_collision_compose() {
        let space = GridSpace::new(10, 10);
        let col = GridCollision::with_blocked([GridState::new(6, 5)]);
        let free = free_successors(&space, &col, &GridState::new(5, 5));

        assert_eq!(free.len(), 3); // the blocked (6,5) is filtered out
        assert!(!free.contains(&GridState::new(6, 5)));
    }
}
