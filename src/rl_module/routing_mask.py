"""Versioned action-mask semantics for masked routing checkpoints."""

FRONTIER_RESTRICTED_EDGES_V1 = "frontier_restricted_edges.v1"
FRONTIER_RESTRICTED_EDGES_V2 = "frontier_restricted_edges.v2"

LEGACY_MASK_SEMANTICS = FRONTIER_RESTRICTED_EDGES_V1
DEFAULT_NEW_MASK_SEMANTICS = FRONTIER_RESTRICTED_EDGES_V2
SUPPORTED_MASK_SEMANTICS = frozenset(
    {
        FRONTIER_RESTRICTED_EDGES_V1,
        FRONTIER_RESTRICTED_EDGES_V2,
    }
)


def normalize_mask_semantics(
    mask_semantics: str | None,
    *,
    default: str = LEGACY_MASK_SEMANTICS,
) -> str:
    resolved_semantics = default if mask_semantics is None else mask_semantics
    if resolved_semantics not in SUPPORTED_MASK_SEMANTICS:
        raise ValueError(
            "Unsupported routing mask semantics "
            f"{resolved_semantics!r}; expected one of {sorted(SUPPORTED_MASK_SEMANTICS)}"
        )
    return resolved_semantics
