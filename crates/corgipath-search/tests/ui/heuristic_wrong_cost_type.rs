//! A heuristic producing a different cost *type* than the space must not compile.
//! The grid's cost is `FixedPoint<100>`; this heuristic yields `u32`.

use corgipath_core::{GridSpace, GridState, NoCollision};
use corgipath_search::{astar, Unbounded};

fn main() {
    let space = GridSpace::new(5, 5);
    let (start, goal) = (GridState::new(0, 0), GridState::new(4, 4));
    let wrong_cost = |_s: &GridState| 0u32;
    let _ = astar(
        &space,
        &NoCollision,
        start,
        goal,
        &wrong_cost,
        &mut (),
        &mut Unbounded,
    );
}
