//! # corgipath-core
//!
//! Foundational contracts shared by every corgipath regime (single, centralized,
//! decentralized) and independent of agent count.
//!
//! The crate is intentionally small and dependency-light: it defines the *ports*
//! that concrete components plug into, expressed as traits so that compatibility
//! is enforced by the compiler rather than by runtime guards.
//!
//! - [`space`] — the [`SearchSpace`], [`Collision`], and [`Cost`] contracts.
//! - [`fixed`] — [`FixedPoint`], a named user-tunable fixed-point cost.
//! - [`grid`] — a batteries-included reference implementation (4-/8-connected grid).
//! - [`observe`] — the passive observer + serializable trace event stream every
//!   visualization backend plugs into.
//! - [`viz`] — a dependency-free ASCII view of a grid search (shared by the TUI
//!   and by snapshot tests).

pub mod fixed;
pub mod grid;
pub mod observe;
pub mod space;
pub mod viz;

pub use observe::{Observer, Outcome, RecordedEvent, Recorder, SearchEvent};
pub use space::{Collision, Cost, NoCollision, SearchSpace};

pub use fixed::FixedPoint;
pub use grid::{Connectivity, GridCollision, GridCost, GridSpace, GridState, CARDINAL, DIAGONAL};
pub use viz::{CellKind, GridView};
