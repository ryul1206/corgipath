//! Heuristics: an estimate of the remaining cost from a state to the goal.
//!
//! A heuristic is **goal-relative** — it is built already knowing the goal, so
//! [`Heuristic::estimate`] takes only the current state. Any closure `Fn(&S) -> C`
//! is a heuristic, so simple cases need no new type.
//!
//! If a heuristic is **admissible** (never overestimates the true remaining
//! cost), A\* returns optimal paths. corgipath cannot enforce admissibility, so
//! it is the caller's responsibility. To make the *right* choice unmissable, ask
//! the space for it: [`grid_heuristic`] returns the heuristic matching a grid's
//! connectivity, so you can never pair (say) Manhattan with an 8-connected grid.
//!
//! Grid heuristics return [`GridCost`] (`FixedPoint<100>`): one orthogonal step is
//! `100`, one diagonal `141`. So `manhattan` counts `100` per cell, `octile` uses
//! `100·max + 41·min`, etc.

use corgipath_core::{Connectivity, Cost, GridCost, GridSpace, GridState};

/// An estimate of the cost remaining from a state to the goal.
pub trait Heuristic<S, C> {
    /// Estimated cost-to-go from `state`.
    fn estimate(&self, state: &S) -> C;
}

/// Every `Fn(&S) -> C` closure is a heuristic.
impl<S, C, F: Fn(&S) -> C> Heuristic<S, C> for F {
    #[inline]
    fn estimate(&self, state: &S) -> C {
        self(state)
    }
}

/// The zero heuristic, `h ≡ 0`. Turns the A\* engine into Dijkstra.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroHeuristic;

impl<S, C: Cost> Heuristic<S, C> for ZeroHeuristic {
    #[inline]
    fn estimate(&self, _state: &S) -> C {
        C::zero()
    }
}

fn deltas(a: &GridState, goal: &GridState) -> (u32, u32) {
    ((a.x - goal.x).unsigned_abs(), (a.y - goal.y).unsigned_abs())
}

/// Manhattan distance to `goal`. Admissible for a **4-connected** grid.
/// Overestimates diagonals, so it is *inadmissible* for 8-connected grids.
pub fn manhattan(goal: GridState) -> impl Heuristic<GridState, GridCost> {
    move |s: &GridState| {
        let (dx, dy) = deltas(s, &goal);
        GridCost::whole(dx + dy)
    }
}

/// Chebyshev distance to `goal`: `max(|dx|, |dy|)` orthogonal steps. Admissible
/// for 8-connected grids (a looser bound than octile, never an overestimate).
pub fn chebyshev(goal: GridState) -> impl Heuristic<GridState, GridCost> {
    move |s: &GridState| {
        let (dx, dy) = deltas(s, &goal);
        GridCost::whole(dx.max(dy))
    }
}

/// Octile distance to `goal`: the exact shortest 8-connected distance with
/// orthogonal cost `100` and diagonal cost `141` (`100·max + 41·min`). Admissible
/// and consistent for 8-connected grids.
pub fn octile(goal: GridState) -> impl Heuristic<GridState, GridCost> {
    move |s: &GridState| {
        let (dx, dy) = deltas(s, &goal);
        let (lo, hi) = (dx.min(dy), dx.max(dy));
        GridCost::raw(100 * hi + 41 * lo)
    }
}

/// The admissible heuristic matching a grid's connectivity, as an enum so it can
/// be returned from a runtime choice. Built via [`grid_heuristic`].
#[derive(Debug, Clone, Copy)]
pub enum GridHeuristic {
    /// 4-connected: Manhattan.
    Manhattan(GridState),
    /// 8-connected, exact: octile.
    Octile(GridState),
    /// 8-connected, looser: Chebyshev.
    Chebyshev(GridState),
}

impl Heuristic<GridState, GridCost> for GridHeuristic {
    fn estimate(&self, state: &GridState) -> GridCost {
        match self {
            GridHeuristic::Manhattan(g) => manhattan(*g).estimate(state),
            GridHeuristic::Octile(g) => octile(*g).estimate(state),
            GridHeuristic::Chebyshev(g) => chebyshev(*g).estimate(state),
        }
    }
}

/// The admissible heuristic for `space`'s connectivity. Because the space chooses,
/// you can never pair an inadmissible heuristic (e.g. Manhattan on an 8-connected
/// grid) by mistake — while [`astar`](crate::astar) stays open to any custom
/// heuristic for those who want one.
pub fn grid_heuristic(space: &GridSpace, goal: GridState) -> GridHeuristic {
    match space.connectivity() {
        Connectivity::Four => GridHeuristic::Manhattan(goal),
        Connectivity::Eight => GridHeuristic::Octile(goal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use corgipath_core::FixedPoint;

    fn g(x: i32, y: i32) -> GridState {
        GridState::new(x, y)
    }

    #[test]
    fn zero_heuristic_is_zero() {
        let h = ZeroHeuristic;
        assert_eq!(
            Heuristic::<GridState, GridCost>::estimate(&h, &g(3, 4)),
            FixedPoint(0)
        );
    }

    #[test]
    fn manhattan_counts_hundred_per_cell() {
        let h = manhattan(g(0, 0));
        assert_eq!(h.estimate(&g(3, 4)), FixedPoint(700));
    }

    #[test]
    fn chebyshev_is_the_larger_axis() {
        let h = chebyshev(g(0, 0));
        assert_eq!(h.estimate(&g(3, 4)), FixedPoint(400));
    }

    #[test]
    fn octile_mixes_cardinal_and_diagonal() {
        let h = octile(g(0, 0));
        // dx=3, dy=4 -> 3 diagonals (141) + 1 cardinal (100) = 523
        assert_eq!(h.estimate(&g(3, 4)), FixedPoint(100 * 4 + 41 * 3));
    }

    #[test]
    fn grid_heuristic_matches_connectivity() {
        let four = GridSpace::new(5, 5);
        let eight = GridSpace::new(5, 5).with_connectivity(Connectivity::Eight);
        assert!(matches!(
            grid_heuristic(&four, g(4, 4)),
            GridHeuristic::Manhattan(_)
        ));
        assert!(matches!(
            grid_heuristic(&eight, g(4, 4)),
            GridHeuristic::Octile(_)
        ));
    }
}
