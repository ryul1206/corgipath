//! [`FixedPoint`] — a named, user-tunable fixed-point integer cost.
//!
//! Fixed-point arithmetic stores fractional values as integers by multiplying by
//! a fixed *scale* (like keeping money in integer cents, or distance in integer
//! millimeters). `FixedPoint<SCALE>` represents `1.0` as `SCALE`: so in
//! `FixedPoint<100>`, one whole unit is `100` and `√2 ≈ 1.41` is `141`.
//!
//! The scale lives in the **type**, so `FixedPoint<1>` and `FixedPoint<100>` are
//! distinct types — a heuristic in one scale will not compile against a space in
//! another, catching unit mismatches structurally. The scale is a free parameter,
//! so users pick their own resolution (and can ignore this type entirely and
//! implement [`Cost`](crate::Cost) for a metric of their own).

use crate::space::Cost;
use core::ops::Add;

/// A fixed-point integer cost where `1.0 == SCALE`.
///
/// `+` adds the underlying integers and `Ord` compares them, so the scale is
/// purely a type-level tag plus a convenience for converting whole units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FixedPoint<const SCALE: u32>(pub u32);

impl<const SCALE: u32> FixedPoint<SCALE> {
    /// The scale factor (units representing `1.0`).
    pub const SCALE: u32 = SCALE;

    /// The cost of one whole unit (`1.0`).
    pub const ONE: Self = FixedPoint(SCALE);

    /// Wrap a raw fixed-point value (already scaled).
    pub const fn raw(value: u32) -> Self {
        FixedPoint(value)
    }

    /// `n` whole units, i.e. `n * SCALE`.
    pub const fn whole(n: u32) -> Self {
        FixedPoint(n * SCALE)
    }

    /// The underlying raw (scaled) integer.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl<const SCALE: u32> Add for FixedPoint<SCALE> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        FixedPoint(self.0 + rhs.0)
    }
}

impl<const SCALE: u32> Cost for FixedPoint<SCALE> {
    #[inline]
    fn zero() -> Self {
        FixedPoint(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_units_scale() {
        assert_eq!(FixedPoint::<100>::whole(3), FixedPoint(300));
        assert_eq!(FixedPoint::<100>::ONE, FixedPoint(100));
        assert_eq!(FixedPoint::<1>::whole(3), FixedPoint(3));
    }

    #[test]
    fn behaves_as_a_cost() {
        let a = FixedPoint::<100>::whole(1); // 100
        let b = FixedPoint::<100>::raw(141); // √2
        assert_eq!(a + b, FixedPoint(241));
        assert_eq!(FixedPoint::<100>::zero(), FixedPoint(0));
        assert!(b > a); // 141 > 100
    }
}
