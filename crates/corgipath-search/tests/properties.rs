//! Property tests: invariants that must hold over *random* grids, obstacle sets,
//! and connectivity (4- or 8-connected), not just hand-picked cases. These pin the
//! algorithms' guarantees — optimality, path validity, Dijkstra/A\* agreement,
//! determinism — across both motion models.

use corgipath_core::{
    Collision, Connectivity, GridCollision, GridCost, GridSpace, GridState, Outcome,
};
use corgipath_search::{astar, dijkstra, grid_heuristic, Path, Unbounded};
use proptest::prelude::*;

fn cell(x: i32, y: i32) -> GridState {
    GridState::new(x, y)
}

/// A valid path: starts at `start`, ends at `goal`, every step moves to an
/// in-bounds, obstacle-free neighbor allowed by the grid's connectivity, and its
/// cost equals the sum of the per-step costs (`100` cardinal, `141` diagonal).
fn check_path(
    space: &GridSpace,
    col: &GridCollision,
    start: GridState,
    goal: GridState,
    path: &Path<GridState, GridCost>,
) -> Result<(), TestCaseError> {
    prop_assert_eq!(path.states.first(), Some(&start));
    prop_assert_eq!(path.states.last(), Some(&goal));

    let mut expected_cost = 0u32;
    for window in path.states.windows(2) {
        let (a, b) = (window[0], window[1]);
        let dx = (a.x - b.x).abs();
        let dy = (a.y - b.y).abs();
        // Chebyshev adjacency: one of the (up to) 8 neighbors.
        prop_assert_eq!(dx.max(dy), 1, "non-adjacent step {:?} -> {:?}", a, b);

        if dx == 1 && dy == 1 {
            // A diagonal step is only legal on an 8-connected grid.
            prop_assert_eq!(
                space.connectivity(),
                Connectivity::Eight,
                "diagonal step on a 4-connected grid"
            );
            expected_cost += 141;
        } else {
            expected_cost += 100;
        }
    }
    prop_assert_eq!(path.cost.get(), expected_cost, "cost disagrees with steps");

    for s in &path.states {
        prop_assert!(space.in_bounds(s), "out of bounds: {:?}", s);
        prop_assert!(!col.is_colliding(s), "path enters obstacle: {:?}", s);
    }
    Ok(())
}

prop_compose! {
    /// A random scenario: dimensions, connectivity, obstacles, and start/goal that
    /// are guaranteed in-bounds and not blocked.
    fn scenario()(
        w in 2i32..=12,
        h in 2i32..=12,
        eight in any::<bool>(),
        raw_obstacles in prop::collection::vec((0i32..12, 0i32..12), 0..48),
        s in (0i32..12, 0i32..12),
        g in (0i32..12, 0i32..12),
    ) -> (GridSpace, GridCollision, GridState, GridState) {
        let connectivity = if eight { Connectivity::Eight } else { Connectivity::Four };
        let start = cell(s.0 % w, s.1 % h);
        let goal = cell(g.0 % w, g.1 % h);
        let blocked: Vec<GridState> = raw_obstacles
            .into_iter()
            .map(|(x, y)| cell(x % w, y % h))
            .filter(|c| *c != start && *c != goal)
            .collect();
        (
            GridSpace::new(w, h).with_connectivity(connectivity),
            GridCollision::with_blocked(blocked),
            start,
            goal,
        )
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Dijkstra and A\* must agree on reachability and optimal cost, and any path
    /// they return must be valid. With the space's own (admissible) heuristic, A\*
    /// expands no more nodes than Dijkstra — on both connectivities.
    #[test]
    fn astar_agrees_with_dijkstra((space, col, start, goal) in scenario()) {
        let d = dijkstra(&space, &col, start, goal, &mut (), &mut Unbounded);
        let a = astar(
            &space, &col, start, goal,
            &grid_heuristic(&space, goal), &mut (), &mut Unbounded,
        );

        prop_assert_eq!(d.outcome, a.outcome);
        match a.outcome {
            Outcome::Solved => {
                let dp = d.path.as_ref().unwrap();
                let ap = a.path.as_ref().unwrap();
                prop_assert_eq!(ap.cost, dp.cost); // both optimal
                check_path(&space, &col, start, goal, ap)?;
                check_path(&space, &col, start, goal, dp)?;
                prop_assert!(
                    a.stats.expanded <= d.stats.expanded,
                    "A* expanded {} > Dijkstra {}",
                    a.stats.expanded,
                    d.stats.expanded
                );
            }
            _ => {
                prop_assert!(a.path.is_none());
                prop_assert!(d.path.is_none());
            }
        }
    }

    /// The same query run twice yields identical results (deterministic
    /// tie-breaking).
    #[test]
    fn search_is_deterministic((space, col, start, goal) in scenario()) {
        let r1 = astar(
            &space, &col, start, goal,
            &grid_heuristic(&space, goal), &mut (), &mut Unbounded,
        );
        let r2 = astar(
            &space, &col, start, goal,
            &grid_heuristic(&space, goal), &mut (), &mut Unbounded,
        );
        prop_assert_eq!(r1.path, r2.path);
        prop_assert_eq!(r1.stats, r2.stats);
        prop_assert_eq!(r1.outcome, r2.outcome);
    }
}
