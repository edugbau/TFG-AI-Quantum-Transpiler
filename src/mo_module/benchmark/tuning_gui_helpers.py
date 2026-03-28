import numpy as np


def _parse_manual_ref_point(text, expected_dims=2):
    raw_parts = [part.strip() for part in text.split(',') if part.strip()]
    if not raw_parts:
        raise ValueError("El ref_point manual es obligatorio.")
    if len(raw_parts) != expected_dims:
        raise ValueError(f"El ref_point manual debe tener {expected_dims} valores.")

    try:
        values = tuple(float(part) for part in raw_parts)
    except ValueError as exc:
        raise ValueError("El ref_point manual debe contener valores float separados por comas.") from exc

    if not np.all(np.isfinite(values)):
        raise ValueError("El ref_point manual debe contener valores finitos.")

    return values


def _format_ref_point_display(ref_point, mode):
    if ref_point is None:
        return "calibrated pending (warm-up automatic)" if mode == "calibrated" else "manual required"

    coords = ", ".join(f"{float(value):.3f}" for value in ref_point)
    return f"{mode} [{coords}]"


def _format_ref_point_mode_help(mode):
    if mode == "manual":
        return "Modo manual: omite el warm-up y exige ref_point manual."
    return "Modo calibrado: usa warm-up automatico para fijar el ref_point."


def _resolve_ref_point_display(mode, ref_point=None, manual_ref_point=None):
    if ref_point is not None:
        return _format_ref_point_display(ref_point, mode)
    if mode == "manual" and manual_ref_point is not None:
        return _format_ref_point_display(manual_ref_point, mode)
    return _format_ref_point_display(None, mode)
