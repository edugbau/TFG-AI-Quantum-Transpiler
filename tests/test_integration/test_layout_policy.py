import numpy as np
import pytest

from src.integration import LayoutSelectionPolicy
from src.integration.layout_policy import select_layout_from_mo_result
from src.mo_module.optimizer import OptimizationResult


def test_compromise_policy_returns_valid_layout() -> None:
    result = OptimizationResult(
        pareto_layouts=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        pareto_fitness=np.array([
            [1.0, 10.0],
            [5.0, 5.0],
            [10.0, 1.0],
        ]),
        objective_names=["depth", "error"],
    )

    selected_layout = select_layout_from_mo_result(
        result,
        policy=LayoutSelectionPolicy.COMPROMISE,
    )

    assert selected_layout == result.get_compromise_layout()
    assert selected_layout == [3, 4, 5]
    assert selected_layout != result.get_best_layout(0)
    assert selected_layout != result.get_best_layout(1)


def test_best_on_objective_policy_returns_best_layout_for_objective() -> None:
    result = OptimizationResult(
        pareto_layouts=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        pareto_fitness=np.array([
            [10.0, 0.05],
            [8.0, 0.10],
            [15.0, 0.03],
        ]),
        objective_names=["depth", "error"],
    )

    selected_layout = select_layout_from_mo_result(
        result,
        policy=LayoutSelectionPolicy.BEST_ON_OBJECTIVE,
        objective_index=1,
    )

    assert selected_layout == [6, 7, 8]


def test_unsupported_policy_raises_value_error() -> None:
    result = OptimizationResult(
        pareto_layouts=[[0, 1, 2]],
        pareto_fitness=np.array([[1.0, 2.0]]),
        objective_names=["depth", "error"],
    )

    with pytest.raises(ValueError, match="Unsupported layout selection policy"):
        select_layout_from_mo_result(result, policy="unsupported")
