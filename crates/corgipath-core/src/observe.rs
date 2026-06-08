//! Passive observation of a search's progress.
//!
//! This is the common base every visualization backend plugs into: algorithms
//! emit a stream of [`SearchEvent`]s to an injected [`Observer`], decoupling
//! instrumentation from algorithm logic (no prints/timeouts baked into the loop).
//!
//! Design notes:
//! - Push model: algorithms call [`Observer::observe`]; the provided [`Recorder`]
//!   collects an owned, serializable trace stream.
//! - Generic over the state `S` and cost `C` as free type parameters, so this
//!   module does not depend on the `State`/`SearchSpace` contracts.
//! - Node-level incremental events; a backend reconstructs the frontier by
//!   accumulating them.
//! - Serialization lives on the owned [`RecordedEvent`]/[`Outcome`] only, behind
//!   the `serde` feature. Borrowed [`SearchEvent`]s are never serialized.
//! - Observation is *passive* — it can never alter control flow. Stopping a search
//!   (budget/deadline) is a separate, active concern handled by the algorithms.

use core::fmt;

/// A passive observer of search progress.
///
/// Generic over the state `S` and cost `C` the algorithm uses. The default
/// no-op observer is `()`, which monomorphizes away to nothing — so an
/// un-observed search pays zero cost.
///
/// `observe` takes the event **by value**; the event borrows the state (`&S`)
/// and owns the cheap, `Copy`-like costs, so constructing one allocates nothing.
pub trait Observer<S, C> {
    /// Called once per search event, in emission order.
    fn observe(&mut self, event: SearchEvent<'_, S, C>);
}

/// The no-op observer: ignores every event. Monomorphized to nothing.
impl<S, C> Observer<S, C> for () {
    #[inline]
    fn observe(&mut self, _event: SearchEvent<'_, S, C>) {}
}

/// How a search ended.
///
/// `#[non_exhaustive]` so further stopping reasons can be added later without a
/// breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Outcome {
    /// The goal was reached.
    Solved,
    /// The frontier emptied without reaching the goal (no path exists).
    Exhausted,
    /// An injected budget stopped the search before it concluded.
    DeadlineReached,
}

/// A single progress event, emitted by an algorithm to an [`Observer`].
///
/// Borrows the state so emission is allocation-free even when recording is off.
/// `#[non_exhaustive]` so new event kinds can be added non-breakingly.
#[derive(Debug)]
#[non_exhaustive]
pub enum SearchEvent<'a, S, C> {
    /// Emitted once at the start.
    Started { start: &'a S, goal: &'a S },
    /// A node was popped from the frontier and is being expanded.
    Expanded { state: &'a S, g: C, f: C },
    /// A successor was discovered for the first time and enqueued.
    Generated { state: &'a S, g: C, f: C },
    /// A cheaper path to an already-seen node was found (its `g` decreased).
    Improved { state: &'a S, g: C, f: C },
    /// The goal node was reached.
    GoalReached { state: &'a S, g: C },
    /// Emitted once at the end, carrying the outcome and how many nodes expanded.
    Finished { outcome: Outcome, expanded: u64 },
}

impl<S: Clone, C> SearchEvent<'_, S, C> {
    /// Clone the borrowed state into an owned [`RecordedEvent`]. Costs (`C`) are
    /// moved, not cloned, since the event owns them.
    pub fn into_owned(self) -> RecordedEvent<S, C> {
        match self {
            SearchEvent::Started { start, goal } => RecordedEvent::Started {
                start: start.clone(),
                goal: goal.clone(),
            },
            SearchEvent::Expanded { state, g, f } => RecordedEvent::Expanded {
                state: state.clone(),
                g,
                f,
            },
            SearchEvent::Generated { state, g, f } => RecordedEvent::Generated {
                state: state.clone(),
                g,
                f,
            },
            SearchEvent::Improved { state, g, f } => RecordedEvent::Improved {
                state: state.clone(),
                g,
                f,
            },
            SearchEvent::GoalReached { state, g } => RecordedEvent::GoalReached {
                state: state.clone(),
                g,
            },
            SearchEvent::Finished { outcome, expanded } => {
                RecordedEvent::Finished { outcome, expanded }
            }
        }
    }
}

/// An owned [`SearchEvent`] — the serializable form held by a [`Recorder`].
///
/// Serialization is gated behind the `serde` feature so the core stays
/// dependency-light. This owned form is what Python/Rerun backends consume once
/// bindings land.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum RecordedEvent<S, C> {
    Started { start: S, goal: S },
    Expanded { state: S, g: C, f: C },
    Generated { state: S, g: C, f: C },
    Improved { state: S, g: C, f: C },
    GoalReached { state: S, g: C },
    Finished { outcome: Outcome, expanded: u64 },
}

/// An [`Observer`] that records every event into an owned, replayable,
/// serializable trace stream.
///
/// The recorded `Vec` *is* the "serializable trace event stream" the design docs
/// describe; SVG/TUI/Rerun/Python backends all consume it.
pub struct Recorder<S, C> {
    events: Vec<RecordedEvent<S, C>>,
}

impl<S, C> Recorder<S, C> {
    /// Create an empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// The events recorded so far, in emission order.
    pub fn events(&self) -> &[RecordedEvent<S, C>] {
        &self.events
    }

    /// Consume the recorder, returning the owned event stream.
    pub fn into_events(self) -> Vec<RecordedEvent<S, C>> {
        self.events
    }

    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether no events have been recorded.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Drop all recorded events, reusing the allocation.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

// Manual `Default` (not derived) so it does not impose `S: Default, C: Default`.
impl<S, C> Default for Recorder<S, C> {
    fn default() -> Self {
        Self { events: Vec::new() }
    }
}

impl<S: Clone, C> Observer<S, C> for Recorder<S, C> {
    fn observe(&mut self, event: SearchEvent<'_, S, C>) {
        self.events.push(event.into_owned());
    }
}

impl<S: fmt::Debug, C: fmt::Debug> Recorder<S, C> {
    /// Render the recorded stream as one deterministic line per event.
    ///
    /// Seed for human-readable debugging and for ASCII/insta snapshot tests.
    pub fn to_trace_string(&self) -> String {
        use core::fmt::Write as _;
        let mut out = String::new();
        for ev in &self.events {
            match ev {
                RecordedEvent::Started { start, goal } => {
                    let _ = writeln!(out, "started start={start:?} goal={goal:?}");
                }
                RecordedEvent::Expanded { state, g, f } => {
                    let _ = writeln!(out, "expanded {state:?} g={g:?} f={f:?}");
                }
                RecordedEvent::Generated { state, g, f } => {
                    let _ = writeln!(out, "generated {state:?} g={g:?} f={f:?}");
                }
                RecordedEvent::Improved { state, g, f } => {
                    let _ = writeln!(out, "improved {state:?} g={g:?} f={f:?}");
                }
                RecordedEvent::GoalReached { state, g } => {
                    let _ = writeln!(out, "goal_reached {state:?} g={g:?}");
                }
                RecordedEvent::Finished { outcome, expanded } => {
                    let _ = writeln!(out, "finished {outcome:?} expanded={expanded}");
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny synthetic event sequence, driven into an arbitrary observer.
    /// Mirrors what Dijkstra/A* will emit, so the observer contract is exercised
    /// before any real algorithm exists.
    fn drive<O: Observer<i32, u32>>(obs: &mut O) {
        let (start, a, goal) = (0i32, 1i32, 2i32);
        obs.observe(SearchEvent::Started {
            start: &start,
            goal: &goal,
        });
        obs.observe(SearchEvent::Expanded {
            state: &start,
            g: 0,
            f: 5,
        });
        obs.observe(SearchEvent::Generated {
            state: &a,
            g: 2,
            f: 6,
        });
        obs.observe(SearchEvent::Improved {
            state: &a,
            g: 1,
            f: 5,
        });
        obs.observe(SearchEvent::GoalReached { state: &goal, g: 4 });
        obs.observe(SearchEvent::Finished {
            outcome: Outcome::Solved,
            expanded: 3,
        });
    }

    #[test]
    fn noop_observer_ignores_everything() {
        // The point is that this compiles and runs against the unit observer.
        let mut obs = ();
        drive(&mut obs);
    }

    #[test]
    fn recorder_captures_events_in_order() {
        let mut rec = Recorder::<i32, u32>::new();
        assert!(rec.is_empty());
        drive(&mut rec);

        assert_eq!(rec.len(), 6);
        assert_eq!(
            rec.events(),
            &[
                RecordedEvent::Started { start: 0, goal: 2 },
                RecordedEvent::Expanded {
                    state: 0,
                    g: 0,
                    f: 5
                },
                RecordedEvent::Generated {
                    state: 1,
                    g: 2,
                    f: 6
                },
                RecordedEvent::Improved {
                    state: 1,
                    g: 1,
                    f: 5
                },
                RecordedEvent::GoalReached { state: 2, g: 4 },
                RecordedEvent::Finished {
                    outcome: Outcome::Solved,
                    expanded: 3,
                },
            ]
        );
    }

    #[test]
    fn trace_string_is_deterministic_and_readable() {
        let mut rec = Recorder::<i32, u32>::new();
        drive(&mut rec);
        let expected = "\
started start=0 goal=2
expanded 0 g=0 f=5
generated 1 g=2 f=6
improved 1 g=1 f=5
goal_reached 2 g=4
finished Solved expanded=3
";
        assert_eq!(rec.to_trace_string(), expected);
    }

    #[test]
    fn clear_resets_recorder() {
        let mut rec = Recorder::<i32, u32>::new();
        drive(&mut rec);
        rec.clear();
        assert!(rec.is_empty());
        assert_eq!(rec.to_trace_string(), "");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn recorded_events_roundtrip_through_json() {
        let mut rec = Recorder::<i32, u32>::new();
        drive(&mut rec);
        let json = serde_json::to_string(rec.events()).unwrap();
        let back: Vec<RecordedEvent<i32, u32>> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, rec.events());
    }
}
