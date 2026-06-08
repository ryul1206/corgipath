# AGENTS.md — corgipath (Rust)

Guidance for contributors and AI agents working in this repo. Keep it current as
the project grows; per-crate `AGENTS.md` files may add local detail.

## What this is

A from-scratch Rust rewrite of corgipath, a customizable path/motion planner for
**MAPF** (Multi-Agent Path Finding). It keeps the *composition philosophy* of the
original Python corgipath: orthogonal, swappable components plug into typed ports
(traits), and the compiler — not a runtime check — enforces which pieces fit
together.

## Design principles

- **Research/benchmark first.** Build a clean, single-process, deterministic,
  reproducible library; a networked multi-robot runtime can come later as another
  implementation of the same coordination trait.
- **Sync core.** Planning and search are pure synchronous CPU code. Async (if any)
  lives only in a network/orchestration shell, never across an FFI boundary.
- **Generic state.** The state is a `SearchSpace::State` associated type, so 2D
  grids, SE(2)+time, lattices, etc. are all first-class. Bindings export concrete
  monomorphized instances (generics can't cross FFI).
- **Open cost.** The `Cost` trait is `Copy + Ord + Add` with an explicit `zero()`,
  so integers, fixed-point, and multi-component (lexicographic) costs all plug in.
- **Clean-room.** Algorithms are derived from papers/pseudocode; do not copy from
  existing implementations.
- **Dependency-light core.** Optional dependencies (serde, visualization) are
  behind feature flags. Visualization is layered: an observer trait + a
  serializable trace-event stream is the common base; backends (terminal TUI,
  SVG snapshots, Rerun, Python) build on it.

## Architecture

```
corgipath-core    : State, SearchSpace, Collision, Map   (agent-count-agnostic)
   ↑
corgipath-search  : Dijkstra, A* (+ constraints/time)    (single-agent for now)
   ↑                         ↑
centralized              decentralized   (future; don't depend on each other)
```

Dependency direction `core → search` is fixed. `search` is designed to accept
constraints and time from the start so multi-agent layers can reuse it.

## Current scope

**Single-agent only.** Algorithms: **Dijkstra** and **A\***, built on the core
trait contracts, test-first.

## Conventions

- **TDD / test-first.** Prefer a failing test before the implementation.
- **No runtime "is_prepared" guards** — encode readiness in types (builder/typestate)
  so an unprepared planner simply doesn't expose `solve()`.
- **Immutable definition / mutable scratch** split: search-space *definitions* are
  immutable, shared, `Send + Sync`; per-search *scratch* is allocated per call.
- **Inject observation & budget**: no timeouts or prints baked into algorithm
  bodies; use the observer trait and an injectable budget/deadline.
- Keep the core dependency-light; gate optional deps (serde, viz) behind features.

## Build & test

```
cargo build
cargo test
cargo fmt && cargo clippy --all-targets
```
