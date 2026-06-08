//! The core ports a search plugs into: [`SearchSpace`] (what states exist and how
//! to move between them), [`Collision`] (is a pose free?), and [`Cost`] (the
//! algebra of edge costs). These are orthogonal, swappable concerns — the
//! compiler enforces which pieces fit together via trait bounds.
//!
//! Design notes:
//! - The state is an associated type ([`SearchSpace::State`]); every method takes
//!   `&self`, so a space is an immutable definition shareable across threads.
//! - Identity is separate from the state ([`SearchSpace::Id`] + [`SearchSpace::id`]),
//!   so continuous spaces can discretize while discrete ones use the state itself.
//! - Successors are pushed into a caller-reused buffer (zero per-expansion alloc).
//! - The cost type is generic via the open [`Cost`] trait, so users can plug their
//!   own metrics (fixed-point, lexicographic tuples, bottleneck, …).
//! - Collision is a separate port, generic over the queried state, with a trivial
//!   [`NoCollision`] for explicit graphs.
//! - Constraints/time are deferred; a generic state can carry them later without
//!   changing these contracts.

use core::hash::Hash;
use core::ops::Add;

/// What a search needs from an edge/path cost: an explicit zero ([`Cost::zero`]),
/// a way to accumulate (`+`), and a total order (`Ord`).
///
/// Implemented for the built-in integers and a fixed-point cost
/// ([`crate::FixedPoint`]). Implement it for your own metric — only `zero` is a
/// method; combine is the `+` operator, so you add a `Cost` impl plus an `Add`
/// impl. This is what richer costs use: a lexicographic key pair (à la D\* Lite),
/// a multi-objective cost, a bottleneck (min-max) cost — each a small struct with
/// `#[derive(Ord)]`, an `Add` carrying its combine semantics, and a `zero`.
///
/// Distinct cost *types* are distinct metrics: a heuristic producing one cost type
/// will not compile against a space using another, so unit/metric mismatches are
/// caught structurally.
///
/// For Dijkstra/A* to be correct, `+` must be associative and **monotonic**
/// (`a + e >= a`), `zero()` its identity, and `Ord` a total order. Those numeric
/// laws are the implementer's responsibility (like a heuristic's admissibility).
pub trait Cost: Copy + Ord + Add<Output = Self> {
    /// The zero cost (additive identity) — a path of no edges.
    fn zero() -> Self;
}

macro_rules! impl_cost_for_int {
    ($($t:ty),* $(,)?) => {$(
        impl Cost for $t {
            #[inline]
            fn zero() -> Self {
                0
            }
        }
    )*};
}
impl_cost_for_int!(u32, u64, usize);

/// What states exist and how to move between them.
///
/// An **immutable definition**: every method takes `&self`, so a `SearchSpace`
/// can be `Send + Sync`, shared across threads, and reused by many searches. The
/// per-search mutable scratch (open/closed sets) lives in the algorithm, not here.
pub trait SearchSpace {
    /// A configuration / node — a grid cell, an SE(2) pose, a lattice node, ….
    type State: Clone;

    /// A hashable identity used to deduplicate states (the closed-set key).
    ///
    /// For a discrete space, `Id` is typically the state itself. For a continuous
    /// space it is a discretization bucket — and *that* mapping is where a space
    /// chooses its discretization (e.g. how to fold an angle like `theta % pi`).
    type Id: Eq + Hash + Clone;

    /// The cost of one edge.
    type Cost: Cost;

    /// The identity of `state`.
    fn id(&self, state: &Self::State) -> Self::Id;

    /// Fill `out` with every successor of `state` and its edge cost.
    ///
    /// Implementations **clear `out` first**, then push, so the caller can pass a
    /// single buffer reused across expansions: after the first few calls its
    /// capacity converges to the max successor count and never reallocates again.
    fn successors(&self, state: &Self::State, out: &mut Vec<(Self::State, Self::Cost)>);
}

/// Is a robot placed at a given state in collision with an obstacle?
///
/// Orthogonal to [`SearchSpace`]: the same motion model can run against different
/// obstacle sets, and a representation has its own checker family (e.g. a grid
/// checker, an SE(2) polygon checker). Generic over the queried state `S` so a
/// single [`NoCollision`] can serve every state type.
///
/// The footprint logic (a body spanning several cells, broad-phase + narrow-phase)
/// lives inside the implementation; the query stays a simple yes/no.
pub trait Collision<S> {
    /// `true` if a robot at `state` overlaps an obstacle.
    fn is_colliding(&self, state: &S) -> bool;
}

/// Everything is free. The trivial checker for explicit graphs whose edges are
/// already valid; usable with any state type.
pub struct NoCollision;

impl<S> Collision<S> for NoCollision {
    #[inline]
    fn is_colliding(&self, _state: &S) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A `Cost` type must support the search's needs: zero, add, compare.
    fn cost_algebra<C: Cost>(a: C, b: C) -> C {
        let zero = C::zero();
        let sum = a + b;
        core::cmp::max(sum, zero)
    }

    #[test]
    fn cost_is_implemented_for_integers() {
        assert_eq!(cost_algebra(2u32, 3u32), 5);
        assert_eq!(cost_algebra(0u64, 0u64), 0);
    }

    #[test]
    fn no_collision_is_never_colliding() {
        let c = NoCollision;
        assert!(!c.is_colliding(&(1, 2)));
        assert!(!c.is_colliding(&"anything"));
    }
}
