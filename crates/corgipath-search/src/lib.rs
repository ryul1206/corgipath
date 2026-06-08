//! # corgipath-search
//!
//! Single-agent graph/state search built on the [`corgipath_core`] contracts.
//! A single best-first engine backs both [`dijkstra`] and [`astar`] (the latter
//! with an injected [`Heuristic`]); searches emit progress through
//! [`corgipath_core::Observer`] and can be bounded by an injected [`Budget`].
//!
//! ```
//! use corgipath_core::{GridSpace, GridState, NoCollision};
//! use corgipath_search::{astar, manhattan, Unbounded};
//!
//! let space = GridSpace::new(10, 10);
//! let (start, goal) = (GridState::new(0, 0), GridState::new(9, 9));
//! let result = astar(
//!     &space, &NoCollision, start, goal,
//!     &manhattan(goal), &mut (), &mut Unbounded,
//! );
//! // Fixed-point cost: 18 cells × 100.
//! assert_eq!(result.path.unwrap().cost.get(), 1800);
//! ```
//!
//! Designed to later accept constraints and time so the same engine can serve as
//! the low-level subroutine for MAPF regimes. Scope now: **single-agent only**.

pub mod budget;
pub mod heuristic;
pub mod search;

pub use budget::{Budget, Control, ExpansionLimit, Progress, Unbounded};
pub use heuristic::{
    chebyshev, grid_heuristic, manhattan, octile, GridHeuristic, Heuristic, ZeroHeuristic,
};
pub use search::{astar, dijkstra, Path, SearchResult};
