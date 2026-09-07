"""Fixed slots for one original plus five augmented copies."""

from __future__ import annotations

# Order of the five augmented variants (original is variant 0).
# easy: three light ops; mild: two mild ops; hard: one heavy op.
AUGMENTED_VARIANT_PLAN: tuple[tuple[str, int], ...] = (
    ("easy", 3),
    ("easy", 3),
    ("mild", 2),
    ("mild", 2),
    ("hard", 1),
)

EXPECTED_N_AUGMENTATIONS = len(AUGMENTED_VARIANT_PLAN)


def plan_for_augmented_variant(variant: int) -> tuple[str, int]:
    """Return ``(strength, operation_count)`` for an augmented variant index."""
    if variant < 1 or variant > len(AUGMENTED_VARIANT_PLAN):
        raise ValueError(
            f"Augmented variant must be between 1 and {len(AUGMENTED_VARIANT_PLAN)}, got {variant}."
        )
    return AUGMENTED_VARIANT_PLAN[variant - 1]
