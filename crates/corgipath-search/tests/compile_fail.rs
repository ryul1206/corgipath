//! Compile-fail tests: incompatible assemblies must **not compile**.
//!
//! This is how corgipath turns "compatibility" into a compile-time guarantee — a
//! trait bound *is* the compatibility marker. Each case in `tests/ui/` pairs a
//! search with a heuristic that disagrees on the state or cost type, and the
//! captured `.stderr` pins that the mismatch is a type error.
//!
//! Regenerate the expected output after an intentional change with:
//! `TRYBUILD=overwrite cargo test -p corgipath-search --test compile_fail`.
//! (The `.stderr` files are tied to the compiler version; refresh them if rustc
//! changes the wording.)

#[test]
fn incompatible_assemblies_do_not_compile() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
