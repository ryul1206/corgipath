//! A heuristic over a different *state* type must not compile. The grid's state is
//! `GridState`; this heuristic estimates over `usize`.

use corgipath_core::{FixedPoint, GridSpace, GridState, NoCollision};
use corgipath_search::{astar, Unbounded};

fn main() {
    let space = GridSpace::new(5, 5);
    let (start, goal) = (GridState::new(0, 0), GridState::new(4, 4));
    let wrong_state = |_s: &usize| FixedPoint::<100>(0);
    let _ = astar(
        &space,
        &NoCollision,
        start,
        goal,
        &wrong_state,
        &mut (),
        &mut Unbounded,
    );
}
