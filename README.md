# CorgiPath

<!--  -->

A customizable path & motion planner for **MAPF (Multi-Agent Path Finding)**,
written in Rust. corgipath is built like a modular synthesizer: orthogonal,
swappable components (search space, collision, planning) plug into typed ports
(Rust traits), and the compiler — not a runtime check — enforces which pieces fit
together.

> Status: **early scaffolding.** Single-agent search (Dijkstra, A\*) is being
> built first. This is a from-scratch Rust rewrite; it inherits only the
> *composition philosophy* of the original Python
> [corgipath](https://github.com/ryul1206/corgipath), not its code.

## Workspace layout

```
crates/
  corgipath-core/     # State, SearchSpace, Collision, Map contracts (regime-agnostic)
  corgipath-search/   # Dijkstra, A* — single-agent search (depends on core)
```

More crates (centralized/decentralized MAPF, C/Python bindings, visualization
backends) arrive as the project grows. The dependency direction `core → search`
is fixed from the start.

## License

corgipath is dual-licensed under either of

- **MIT license** ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>), or
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>),

**at your option.**

### What does "dual-licensed, at your option" mean? (a plain-language note)

If you're new to this: you do **not** have to comply with both licenses. You pick
**whichever one of the two** you prefer, and use corgipath under that license
alone. You can even pick different licenses for different projects.

Why offer both? This is the de-facto standard in the Rust ecosystem:

- **MIT** is short, permissive, and familiar — easy to drop into almost anything.
- **Apache-2.0** says essentially the same thing but adds an **explicit patent
  grant** (contributors can't later sue you over patents covering their
  contributions) and clearer terms for larger/corporate users.

Offering both lets MIT-only projects and Apache-preferring organizations each use
corgipath comfortably. When in doubt, MIT is the simplest choice.

### Contributing

Unless you explicitly state otherwise, any contribution you intentionally submit
for inclusion in corgipath, as defined in the Apache-2.0 license, shall be
dual-licensed as above, without any additional terms or conditions.
