//! Injected search budgets — the *active* counterpart to the passive
//! [`corgipath_core::Observer`]. A budget is queried once per expansion and may
//! stop the search; a stop yields [`corgipath_core::Outcome::DeadlineReached`].
//!
//! Provided now: [`Unbounded`] and the deterministic [`ExpansionLimit`]. A
//! wall-clock `Deadline` (non-deterministic) and combinations may be added later.

/// Whether a search should keep going or stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    /// Keep searching.
    Continue,
    /// Stop now; the search reports `Outcome::DeadlineReached`.
    Stop,
}

/// Running counters describing how far a search has gotten. Passed to
/// [`Budget::check`] so a budget can decide based on progress.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Progress {
    /// Nodes popped and expanded (excludes stale, lazily-deleted pops).
    pub expanded: u64,
    /// Nodes discovered/enqueued for the first time.
    pub generated: u64,
}

/// A budget queried once per expansion. `&mut self` because a budget may hold
/// internal state (e.g. a countdown).
pub trait Budget {
    /// Inspect `progress` and decide whether to continue.
    fn check(&mut self, progress: &Progress) -> Control;
}

/// Never stops. The default — an unbounded search.
#[derive(Debug, Clone, Copy, Default)]
pub struct Unbounded;

impl Budget for Unbounded {
    #[inline]
    fn check(&mut self, _progress: &Progress) -> Control {
        Control::Continue
    }
}

/// Stop after at most `0` expansions — i.e. `ExpansionLimit(n)` allows `n`
/// expansions. Deterministic (independent of wall-clock time), so it preserves
/// reproducibility and is the preferred budget.
#[derive(Debug, Clone, Copy)]
pub struct ExpansionLimit(pub u64);

impl Budget for ExpansionLimit {
    #[inline]
    fn check(&mut self, progress: &Progress) -> Control {
        if progress.expanded >= self.0 {
            Control::Stop
        } else {
            Control::Continue
        }
    }
}
