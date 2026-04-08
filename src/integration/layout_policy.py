from src.integration.contracts import LayoutSelectionPolicy


def select_layout_from_mo_result(
    result,
    *,
    policy,
    objective_index: int = 0,
) -> list[int]:
    try:
        normalized_policy = LayoutSelectionPolicy(policy)
    except ValueError as exc:
        raise ValueError(f"Unsupported layout selection policy: {policy}") from exc

    if normalized_policy == LayoutSelectionPolicy.COMPROMISE:
        return result.get_compromise_layout()

    if normalized_policy == LayoutSelectionPolicy.BEST_ON_OBJECTIVE:
        return result.get_best_layout(objective_index=objective_index)

    raise ValueError(f"Unsupported layout selection policy: {policy}")
