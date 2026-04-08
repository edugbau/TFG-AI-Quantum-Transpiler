import argparse
import json
from dataclasses import asdict

from src.integration.contracts import ScenarioRequest, ScenarioResult
from src.integration.scenarios import (
    run_baseline_scenario,
    run_mo_only_scenario,
    run_mo_rl_scenario,
    run_rl_only_scenario,
)


_SCENARIO_CHOICES = ("Baseline", "MO_Only", "RL_Only", "MO+RL")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=_SCENARIO_CHOICES, required=True)
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--num-qubits", type=int, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rl-model-path")
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


def run_from_args(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    request_kwargs = {
        "scenario_name": args.scenario,
        "circuit_name": args.circuit,
        "num_qubits": args.num_qubits,
        "backend_name": args.backend,
        "rl_model_path": args.rl_model_path,
    }
    if args.seed is not None:
        request_kwargs["seed"] = args.seed
    request = ScenarioRequest(**request_kwargs)
    result = _dispatch(request)
    return _to_serializable_dict(result)


def main() -> int:
    print(json.dumps(run_from_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
