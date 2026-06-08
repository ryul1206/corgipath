//! The single-agent best-first search engine: one implementation shared by
//! [`astar`] and [`dijkstra`] (Dijkstra is A\* with the zero heuristic).
//!
//! All collaborators are passed explicitly: the immutable
//! *definitions* by shared reference (`&space`, `&collision`, `&heuristic`), the
//! per-call mutable side-channels (`&mut observer`, `&mut budget`) by exclusive
//! reference. The per-search *scratch* (open list, cost/parent maps) is allocated
//! inside the call. Because every collaborator is required by the signature,
//! there is no "unprepared" state to guard against at runtime.

use core::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;

use corgipath_core::{Collision, Cost, Observer, Outcome, SearchEvent, SearchSpace};

use crate::budget::{Budget, Control, Progress};
use crate::heuristic::{Heuristic, ZeroHeuristic};

/// A path from start to goal and its total cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Path<S, C> {
    /// Waypoints from start to goal, inclusive.
    pub states: Vec<S>,
    /// Total cost of the path.
    pub cost: C,
}

/// The outcome of a search, with the path (if any) and run statistics.
#[derive(Debug, Clone)]
pub struct SearchResult<S, C> {
    /// The optimal path, or `None` if the search did not reach the goal.
    pub path: Option<Path<S, C>>,
    /// Why the search stopped.
    pub outcome: Outcome,
    /// How much work the search did.
    pub stats: Progress,
}

/// An open-list entry, ordered as a min-heap on `f`, breaking ties by insertion
/// order (`seq`) for deterministic, reproducible results. The
/// `state` rides along but never participates in ordering — so the state type
/// need not be `Ord`.
struct Frontier<S, C> {
    f: C,
    seq: u64,
    g: C,
    state: S,
}

impl<S, C: Ord> PartialEq for Frontier<S, C> {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.seq == other.seq
    }
}
impl<S, C: Ord> Eq for Frontier<S, C> {}
impl<S, C: Ord> PartialOrd for Frontier<S, C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<S, C: Ord> Ord for Frontier<S, C> {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap; reverse so the smallest `f` (then smallest
        // `seq`) is considered greatest and pops first.
        other.f.cmp(&self.f).then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Dijkstra's algorithm: best-first search ordered by path cost. Equivalent to
/// [`astar`] with the [`ZeroHeuristic`].
pub fn dijkstra<Sp, Col, Obs, Bud>(
    space: &Sp,
    collision: &Col,
    start: Sp::State,
    goal: Sp::State,
    observer: &mut Obs,
    budget: &mut Bud,
) -> SearchResult<Sp::State, Sp::Cost>
where
    Sp: SearchSpace,
    Col: Collision<Sp::State>,
    Obs: Observer<Sp::State, Sp::Cost>,
    Bud: Budget,
{
    astar(
        space,
        collision,
        start,
        goal,
        &ZeroHeuristic,
        observer,
        budget,
    )
}

/// A\* search: best-first ordered by `f = g + h`, where `g` is the cost so far
/// and `h` is the injected [`Heuristic`].
///
/// Returns the optimal path when the heuristic is admissible. Handles
/// re-improvement of already-expanded nodes (so it stays correct for admissible
/// but inconsistent heuristics) via cost-map relaxation plus lazy deletion of
/// stale open-list entries.
pub fn astar<Sp, Col, H, Obs, Bud>(
    space: &Sp,
    collision: &Col,
    start: Sp::State,
    goal: Sp::State,
    heuristic: &H,
    observer: &mut Obs,
    budget: &mut Bud,
) -> SearchResult<Sp::State, Sp::Cost>
where
    Sp: SearchSpace,
    Col: Collision<Sp::State>,
    H: Heuristic<Sp::State, Sp::Cost>,
    Obs: Observer<Sp::State, Sp::Cost>,
    Bud: Budget,
{
    let zero = Sp::Cost::zero();
    let goal_id = space.id(&goal);

    observer.observe(SearchEvent::Started {
        start: &start,
        goal: &goal,
    });

    // Per-search scratch.
    let mut best_g: HashMap<Sp::Id, Sp::Cost> = HashMap::new();
    let mut parent: HashMap<Sp::Id, Sp::Id> = HashMap::new();
    let mut state_of: HashMap<Sp::Id, Sp::State> = HashMap::new();
    let mut heap: BinaryHeap<Frontier<Sp::State, Sp::Cost>> = BinaryHeap::new();
    let mut succ_buf: Vec<(Sp::State, Sp::Cost)> = Vec::new();
    let mut stats = Progress::default();
    let mut seq: u64 = 0;

    let start_id = space.id(&start);
    best_g.insert(start_id.clone(), zero);
    state_of.insert(start_id.clone(), start.clone());
    heap.push(Frontier {
        f: zero + heuristic.estimate(&start),
        seq,
        g: zero,
        state: start,
    });
    seq += 1;

    let outcome = loop {
        let Some(top) = heap.pop() else {
            break Outcome::Exhausted;
        };
        let cur_id = space.id(&top.state);

        // Lazy deletion: skip an entry superseded by a cheaper path to the same node.
        if let Some(&best) = best_g.get(&cur_id) {
            if top.g > best {
                continue;
            }
        }

        // Active budget, queried once per (real) expansion.
        if budget.check(&stats) == Control::Stop {
            break Outcome::DeadlineReached;
        }

        stats.expanded += 1;
        observer.observe(SearchEvent::Expanded {
            state: &top.state,
            g: top.g,
            f: top.f,
        });

        if cur_id == goal_id {
            observer.observe(SearchEvent::GoalReached {
                state: &top.state,
                g: top.g,
            });
            break Outcome::Solved;
        }

        space.successors(&top.state, &mut succ_buf);
        for (next, edge) in succ_buf.drain(..) {
            if collision.is_colliding(&next) {
                continue;
            }
            let next_id = space.id(&next);
            let tentative = top.g + edge;

            let is_new = !best_g.contains_key(&next_id);
            let improves = match best_g.get(&next_id) {
                Some(&best) => tentative < best,
                None => true,
            };
            if !improves {
                continue;
            }

            best_g.insert(next_id.clone(), tentative);
            parent.insert(next_id.clone(), cur_id.clone());
            state_of.insert(next_id.clone(), next.clone());
            let f = tentative + heuristic.estimate(&next);

            if is_new {
                stats.generated += 1;
                observer.observe(SearchEvent::Generated {
                    state: &next,
                    g: tentative,
                    f,
                });
            } else {
                observer.observe(SearchEvent::Improved {
                    state: &next,
                    g: tentative,
                    f,
                });
            }

            heap.push(Frontier {
                f,
                seq,
                g: tentative,
                state: next,
            });
            seq += 1;
        }
    };

    let path = if outcome == Outcome::Solved {
        let mut states = Vec::new();
        let mut id = goal_id.clone();
        loop {
            let s = state_of
                .get(&id)
                .expect("reached state must have been recorded")
                .clone();
            states.push(s);
            match parent.get(&id) {
                Some(p) => id = p.clone(),
                None => break,
            }
        }
        states.reverse();
        let cost = best_g
            .get(&goal_id)
            .copied()
            .expect("solved goal must have a cost");
        Some(Path { states, cost })
    } else {
        None
    };

    observer.observe(SearchEvent::Finished {
        outcome,
        expanded: stats.expanded,
    });

    SearchResult {
        path,
        outcome,
        stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::{ExpansionLimit, Unbounded};
    use crate::heuristic::{grid_heuristic, manhattan, octile};
    use corgipath_core::{
        Connectivity, GridCollision, GridSpace, GridState, NoCollision, Recorder, SearchSpace,
    };

    fn at(x: i32, y: i32) -> GridState {
        GridState::new(x, y)
    }

    #[test]
    fn dijkstra_finds_shortest_path_on_open_grid() {
        let space = GridSpace::new(5, 5);
        let r = dijkstra(
            &space,
            &NoCollision,
            at(0, 0),
            at(4, 4),
            &mut (),
            &mut Unbounded,
        );
        let path = r.path.expect("path exists");
        assert_eq!(r.outcome, Outcome::Solved);
        assert_eq!(path.cost.get(), 800); // (4 right + 4 up) × 100
        assert_eq!(path.states.len(), 9); // inclusive of start and goal
        assert_eq!(path.states.first(), Some(&at(0, 0)));
        assert_eq!(path.states.last(), Some(&at(4, 4)));
    }

    #[test]
    fn obstacles_force_a_detour() {
        // A vertical wall x=2 for y in 0..=3, leaving a gap at (2,4).
        let space = GridSpace::new(5, 6);
        let wall = (0..=3).map(|y| at(2, y));
        let col = GridCollision::with_blocked(wall);

        let open = dijkstra(
            &space,
            &NoCollision,
            at(0, 0),
            at(4, 0),
            &mut (),
            &mut Unbounded,
        )
        .path
        .unwrap();
        let detour = dijkstra(&space, &col, at(0, 0), at(4, 0), &mut (), &mut Unbounded)
            .path
            .unwrap();

        assert_eq!(open.cost.get(), 400); // straight line × 100
        assert!(detour.cost > open.cost); // must go around the wall
        assert!(detour
            .states
            .iter()
            .all(|s| !(s.x == 2 && (0..=3).contains(&s.y))));
    }

    #[test]
    fn astar_matches_dijkstra_cost_with_no_more_expansions() {
        let space = GridSpace::new(20, 20);
        let (start, goal) = (at(0, 0), at(19, 19));

        let d = dijkstra(&space, &NoCollision, start, goal, &mut (), &mut Unbounded);
        let a = astar(
            &space,
            &NoCollision,
            start,
            goal,
            &manhattan(goal),
            &mut (),
            &mut Unbounded,
        );

        assert_eq!(a.path.unwrap().cost, d.path.unwrap().cost); // both optimal
        assert!(
            a.stats.expanded <= d.stats.expanded,
            "A* expanded {} vs Dijkstra {}",
            a.stats.expanded,
            d.stats.expanded
        );
    }

    #[test]
    fn zero_heuristic_makes_astar_equal_dijkstra() {
        use crate::heuristic::ZeroHeuristic;
        let space = GridSpace::new(8, 8);
        let (start, goal) = (at(0, 0), at(7, 5));

        let d = dijkstra(&space, &NoCollision, start, goal, &mut (), &mut Unbounded);
        let a = astar(
            &space,
            &NoCollision,
            start,
            goal,
            &ZeroHeuristic,
            &mut (),
            &mut Unbounded,
        );
        assert_eq!(a.stats.expanded, d.stats.expanded);
        assert_eq!(a.path.unwrap().cost, d.path.unwrap().cost);
    }

    #[test]
    fn no_path_returns_exhausted() {
        // Wall off the goal corner completely.
        let space = GridSpace::new(5, 5);
        let col = GridCollision::with_blocked([at(3, 4), at(4, 3)]);
        let r = dijkstra(&space, &col, at(0, 0), at(4, 4), &mut (), &mut Unbounded);
        assert_eq!(r.outcome, Outcome::Exhausted);
        assert!(r.path.is_none());
    }

    #[test]
    fn expansion_limit_yields_deadline_reached() {
        let space = GridSpace::new(50, 50);
        let mut budget = ExpansionLimit(5);
        let r = dijkstra(
            &space,
            &NoCollision,
            at(0, 0),
            at(49, 49),
            &mut (),
            &mut budget,
        );
        assert_eq!(r.outcome, Outcome::DeadlineReached);
        assert!(r.path.is_none());
        assert_eq!(r.stats.expanded, 5);
    }

    #[test]
    fn start_equals_goal_is_a_single_node_path() {
        let space = GridSpace::new(5, 5);
        let r = dijkstra(
            &space,
            &NoCollision,
            at(2, 2),
            at(2, 2),
            &mut (),
            &mut Unbounded,
        );
        let path = r.path.unwrap();
        assert_eq!(path.cost.get(), 0);
        assert_eq!(path.states, vec![at(2, 2)]);
    }

    #[test]
    fn eight_connected_takes_the_diagonal_shortcut() {
        let space = GridSpace::new(8, 8).with_connectivity(Connectivity::Eight);
        let (start, goal) = (at(0, 0), at(3, 3));
        let r = astar(
            &space,
            &NoCollision,
            start,
            goal,
            &octile(goal),
            &mut (),
            &mut Unbounded,
        );
        let path = r.path.unwrap();
        // Three diagonal steps of 141 each, instead of six cardinal steps of 100.
        assert_eq!(path.cost.get(), 3 * 141);
        assert_eq!(path.states, vec![at(0, 0), at(1, 1), at(2, 2), at(3, 3)]);
    }

    #[test]
    fn grid_heuristic_drives_optimal_search_on_both_connectivities() {
        for conn in [Connectivity::Four, Connectivity::Eight] {
            let space = GridSpace::new(12, 12).with_connectivity(conn);
            let (start, goal) = (at(0, 0), at(11, 7));
            let a = astar(
                &space,
                &NoCollision,
                start,
                goal,
                &grid_heuristic(&space, goal),
                &mut (),
                &mut Unbounded,
            );
            let d = dijkstra(&space, &NoCollision, start, goal, &mut (), &mut Unbounded);
            // The auto-selected heuristic is admissible, so A* matches Dijkstra's
            // optimal cost — and never expands more nodes.
            assert_eq!(a.path.unwrap().cost, d.path.unwrap().cost);
            assert!(a.stats.expanded <= d.stats.expanded);
        }
    }

    // A weighted graph with a cheap route discovered late — exercises non-unit
    // costs, the `Improved` event, and lazy deletion of the stale entry.
    struct WeightedGraph {
        adj: Vec<Vec<(usize, u32)>>,
    }
    impl SearchSpace for WeightedGraph {
        type State = usize;
        type Id = usize;
        type Cost = u32;
        fn id(&self, state: &usize) -> usize {
            *state
        }
        fn successors(&self, state: &usize, out: &mut Vec<(usize, u32)>) {
            out.clear();
            out.extend_from_slice(&self.adj[*state]);
        }
    }

    #[test]
    fn weighted_graph_relaxes_and_lazily_deletes() {
        // 0 -> X costs 10; 0 -> Y costs 1; Y -> X costs 1; X -> G costs 1.
        // Optimal: 0 -> Y -> X -> G = 3 (not the direct 0 -> X -> G = 11).
        let (s, y, x, g) = (0usize, 1, 2, 3);
        let graph = WeightedGraph {
            adj: vec![
                vec![(x, 10), (y, 1)], // 0
                vec![(x, 1)],          // Y
                vec![(g, 1)],          // X
                vec![],                // G
            ],
        };
        let mut rec = Recorder::<usize, u32>::new();
        let r = dijkstra(&graph, &NoCollision, s, g, &mut rec, &mut Unbounded);

        let path = r.path.unwrap();
        assert_eq!(path.cost, 3);
        assert_eq!(path.states, vec![s, y, x, g]);
        // S, Y, X, G expanded exactly once each: the stale X(g=10) entry is
        // popped and skipped, never re-expanded.
        assert_eq!(r.stats.expanded, 4);
        // X was first generated at g=10, then improved to g=2.
        let improved_x = rec.events().iter().any(|e| {
            matches!(
                e,
                corgipath_core::RecordedEvent::Improved { state, g: 2, .. } if *state == x
            )
        });
        assert!(improved_x, "expected X to be improved to g=2");
    }

    #[test]
    fn event_stream_starts_and_finishes() {
        use corgipath_core::RecordedEvent;
        let space = GridSpace::new(3, 1);
        let mut rec = Recorder::new();
        let r = dijkstra(
            &space,
            &NoCollision,
            at(0, 0),
            at(2, 0),
            &mut rec,
            &mut Unbounded,
        );
        assert_eq!(r.outcome, Outcome::Solved);

        let ev = rec.events();
        assert!(matches!(ev.first(), Some(RecordedEvent::Started { .. })));
        assert!(matches!(
            ev.last(),
            Some(RecordedEvent::Finished {
                outcome: Outcome::Solved,
                ..
            })
        ));
        assert!(ev
            .iter()
            .any(|e| matches!(e, RecordedEvent::GoalReached { .. })));
    }
}
