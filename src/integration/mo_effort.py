from dataclasses import dataclass
from typing import Iterable


MIN_CUSTOM_MO_POPULATION_SIZE = 4
DEFAULT_MO_POPULATION_SIZE = 30
DEFAULT_MO_N_GENERATIONS = 50


@dataclass(frozen=True)
class EffectiveMoSettings:
    mo_use_quick: bool
    mo_population_size: int
    mo_n_generations: int


_AUTO_MO_EFFORT_TIERS = (
    (7, EffectiveMoSettings(True, DEFAULT_MO_POPULATION_SIZE, DEFAULT_MO_N_GENERATIONS)),
    (10, EffectiveMoSettings(False, 60, 120)),
    (14, EffectiveMoSettings(False, 80, 160)),
    (float("inf"), EffectiveMoSettings(False, 100, 220)),
)


def resolve_effective_mo_settings(
    *,
    qubit_count: int,
    mo_effort_mode: str,
    mo_use_quick: bool | None = None,
    mo_population_size: int | None = None,
    mo_n_generations: int | None = None,
) -> EffectiveMoSettings:
    if type(qubit_count) is not int:
        raise ValueError(f"Unsupported qubit count for MO effort resolution: {qubit_count}")

    if qubit_count <= 0:
        raise ValueError(f"Unsupported qubit count for MO effort resolution: {qubit_count}")

    if mo_effort_mode == "custom":
        if type(mo_use_quick) is not bool:
            raise ValueError("Custom MO effort requires mo_use_quick to be a bool")
        if type(mo_population_size) is not int or mo_population_size < MIN_CUSTOM_MO_POPULATION_SIZE:
            raise ValueError(
                "Custom MO effort requires mo_population_size to be an integer >= "
                f"{MIN_CUSTOM_MO_POPULATION_SIZE}"
            )
        if type(mo_n_generations) is not int or mo_n_generations <= 0:
            raise ValueError("Custom MO effort requires mo_n_generations to be a positive integer")

        return EffectiveMoSettings(
            mo_use_quick=mo_use_quick,
            mo_population_size=mo_population_size,
            mo_n_generations=mo_n_generations,
        )

    if mo_effort_mode != "auto":
        raise ValueError(f"Unsupported MO effort mode: {mo_effort_mode}")

    for max_qubits, settings in _AUTO_MO_EFFORT_TIERS:
        if qubit_count <= max_qubits:
            return settings

    raise ValueError(f"Unsupported qubit count for MO effort resolution: {qubit_count}")


def build_auto_mo_effort_preview(qubit_counts: Iterable[int]) -> list[tuple[int, EffectiveMoSettings]]:
    preview: list[tuple[int, EffectiveMoSettings]] = []
    seen_qubit_counts: set[int] = set()

    for qubit_count in qubit_counts:
        settings = resolve_effective_mo_settings(qubit_count=qubit_count, mo_effort_mode="auto")

        if qubit_count in seen_qubit_counts:
            continue
        seen_qubit_counts.add(qubit_count)
        preview.append((qubit_count, settings))

    return preview
