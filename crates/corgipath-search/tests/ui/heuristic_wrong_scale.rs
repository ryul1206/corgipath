//! A heuristic in a different fixed-point *scale* must not compile. The grid uses
//! `FixedPoint<100>`; this heuristic yields `FixedPoint<1>` — a distinct metric.

use corgipath_core::{FixedPoint, GridSpace, GridState, NoCollision};
use corgipath_search::{astar, Unbounded};

fn main() {
    let space = GridSpace::new(5, 5);
    let (start, goal) = (GridState::new(0, 0), GridState::new(4, 4));
    let wrong_scale = |_s: &GridState| FixedPoint::<1>(0);
    let _ = astar(
        &space,
        &NoCollision,
        start,
        goal,
        &wrong_scale,
        &mut (),
        &mut Unbounded,
    );
}
