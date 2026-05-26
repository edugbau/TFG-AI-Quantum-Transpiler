import argparse
import json
from dataclasses import asdict

from src.integration.contracts import CircuitFormat, CircuitSource, ScenarioRequest, ScenarioResult
from src.integration.scenarios import (
    run_baseline_scenario,
    run_mo_only_scenario,
    run_mo_rl_scenario,
    run_rl_only_scenario,
)
from src.integration.verbosity import suppress_output


_SCENARIO_CHOICES = ("Baseline", "MO_Only", "RL_Only", "MO+RL")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=_SCENARIO_CHOICES, required=True)
    parser.add_argument("--circuit")
    parser.add_argument("--num-qubits", type=int)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rl-model-path")
    parser.add_argument(
        "--circuit-source",
        choices=[source.value for source in CircuitSource],
        default=CircuitSource.LIBRARY.value,
    )
    parser.add_argument("--circuit-path")
    parser.add_argument(
        "--circuit-format",
        choices=[circuit_format.value for circuit_format in CircuitFormat],
        default=CircuitFormat.AUTO.value,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _to_serializable_dict(result: ScenarioResult) -> dict:
    if not isinstance(result, ScenarioResult):
        raise TypeError("runner expected ScenarioResult from scenario dispatch")
    return asdict(result)


def _dispatch(request: ScenarioRequest) -> ScenarioResult:
    if request.scenario_name == "Baseline":
        return run_baseline_scenario(request)
    if request.scenario_name == "MO_Only":
        return run_mo_only_scenario(request)
    if request.scenario_name == "RL_Only":
        return run_rl_only_scenario(request)
    if request.scenario_name == "MO+RL":
        return run_mo_rl_scenario(request)
    raise ValueError(f"Unsupported scenario_name: {request.scenario_name}")


def run_from_args(argv: list[str] | None = None, *, verbose: bool | None = None) -> dict:
    args = build_parser().parse_args(argv)
    effective_verbose = args.verbose if verbose is None else verbose
    request_kwargs = {
        "scenario_name": args.scenario,
        "circuit_name": args.circuit,
        "num_qubits": args.num_qubits,
        "backend_name": args.backend,
        "rl_model_path": args.rl_model_path,
        "circuit_source": CircuitSource(args.circuit_source),
        "circuit_path": args.circuit_path,
        "circuit_format": CircuitFormat(args.circuit_format),
    }
    if args.seed is not None:
        request_kwargs["seed"] = args.seed
    request = ScenarioRequest(**request_kwargs)
    with suppress_output(enabled=not effective_verbose):
        result = _dispatch(request)
    return _to_serializable_dict(result)


def main() -> int:
    print(json.dumps(run_from_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
