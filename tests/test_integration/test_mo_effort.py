import pytest

from src.integration.mo_effort import (
    MIN_CUSTOM_MO_POPULATION_SIZE,
    EffectiveMoSettings,
    build_auto_mo_effort_preview,
    resolve_effective_mo_settings,
)


def test_resolve_effective_mo_settings_uses_auto_tiers() -> None:
    assert resolve_effective_mo_settings(qubit_count=7, mo_effort_mode="auto") == EffectiveMoSettings(
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
    )
    assert resolve_effective_mo_settings(qubit_count=8, mo_effort_mode="auto") == EffectiveMoSettings(
        mo_use_quick=False,
        mo_population_size=60,
        mo_n_generations=120,
    )
    assert resolve_effective_mo_settings(qubit_count=10, mo_effort_mode="auto") == EffectiveMoSettings(
        mo_use_quick=False,
        mo_population_size=60,
        mo_n_generations=120,
    )
    assert resolve_effective_mo_settings(qubit_count=11, mo_effort_mode="auto") == EffectiveMoSettings(
        mo_use_quick=False,
        mo_population_size=80,
        mo_n_generations=160,
    )
    assert resolve_effective_mo_settings(qubit_count=12, mo_effort_mode="auto") == EffectiveMoSettings(
        mo_use_quick=False,
        mo_population_size=80,
        mo_n_generations=160,
    )
    assert resolve_effective_mo_settings(qubit_count=14, mo_effort_mode="auto") == EffectiveMoSettings(
        mo_use_quick=False,
        mo_population_size=80,
        mo_n_generations=160,
    )
    assert resolve_effective_mo_settings(qubit_count=15, mo_effort_mode="auto") == EffectiveMoSettings(
        mo_use_quick=False,
        mo_population_size=100,
        mo_n_generations=220,
    )


def test_resolve_effective_mo_settings_returns_custom_values_verbatim() -> None:
    assert resolve_effective_mo_settings(
        qubit_count=12,
        mo_effort_mode="custom",
        mo_use_quick=True,
        mo_population_size=123,
        mo_n_generations=456,
    ) == EffectiveMoSettings(
        mo_use_quick=True,
        mo_population_size=123,
        mo_n_generations=456,
    )


def test_build_auto_mo_effort_preview_collapses_duplicate_qubit_sizes_preserving_order() -> None:
    assert build_auto_mo_effort_preview([12, 7, 12, 15, 8, 7]) == [
        (12, EffectiveMoSettings(False, 80, 160)),
        (7, EffectiveMoSettings(True, 30, 50)),
        (15, EffectiveMoSettings(False, 100, 220)),
        (8, EffectiveMoSettings(False, 60, 120)),
    ]


@pytest.mark.parametrize(
    ("mo_use_quick", "mo_population_size", "mo_n_generations"),
    [
        (None, 10, 20),
        ("true", 10, 20),
        (True, None, 20),
        (True, True, 20),
        (True, False, 20),
        (True, MIN_CUSTOM_MO_POPULATION_SIZE - 3, 20),
        (True, MIN_CUSTOM_MO_POPULATION_SIZE - 2, 20),
        (True, MIN_CUSTOM_MO_POPULATION_SIZE - 1, 20),
        (True, 0, 20),
        (True, -1, 20),
        (True, 10, None),
        (True, 10, True),
        (True, 10, False),
        (True, 10, 0),
        (True, 10, -1),
    ],
)
def test_resolve_effective_mo_settings_rejects_invalid_custom_payloads(
    mo_use_quick: object,
    mo_population_size: object,
    mo_n_generations: object,
) -> None:
    with pytest.raises(ValueError):
        resolve_effective_mo_settings(
            qubit_count=12,
            mo_effort_mode="custom",
            mo_use_quick=mo_use_quick,
            mo_population_size=mo_population_size,
            mo_n_generations=mo_n_generations,
        )


@pytest.mark.parametrize("qubit_count", [0, -1])
def test_resolve_effective_mo_settings_rejects_non_positive_qubit_count(qubit_count: int) -> None:
    with pytest.raises(ValueError):
        resolve_effective_mo_settings(qubit_count=qubit_count, mo_effort_mode="auto")


@pytest.mark.parametrize("qubit_count", [7.5, True])
def test_resolve_effective_mo_settings_rejects_non_integer_qubit_count(qubit_count: object) -> None:
    with pytest.raises(ValueError):
        resolve_effective_mo_settings(qubit_count=qubit_count, mo_effort_mode="auto")


def test_build_auto_mo_effort_preview_rejects_invalid_qubit_count_entries() -> None:
    with pytest.raises(ValueError):
        build_auto_mo_effort_preview([True, 1])


def test_build_auto_mo_effort_preview_validates_each_entry_before_deduplication() -> None:
    with pytest.raises(ValueError):
        build_auto_mo_effort_preview([1, True])


def test_build_auto_mo_effort_preview_accepts_iterables_and_preserves_ordered_uniques() -> None:
    assert build_auto_mo_effort_preview((3, 8, 3)) == [
        (3, EffectiveMoSettings(True, 30, 50)),
        (8, EffectiveMoSettings(False, 60, 120)),
    ]
