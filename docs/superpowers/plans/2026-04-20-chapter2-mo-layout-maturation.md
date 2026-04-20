# Chapter 2 MO Layout Maturation Notes

## Pre-Existing Or Out-Of-Scope Issues Observed During Review

These items were observed during Phase 1 review but are not part of Phase 1 closure. They should be handled in later benchmark/reporting work.

- `src/mo_module/benchmark/analysis.py` `BenchmarkReport.to_dict()` omits HV comparison outputs even though `analyze_results()` computes them and `to_text()` reports them.
- `src/mo_module/benchmark/analysis.py` `CircuitAnalysis.n_seeds` can mislead in mixed empty/non-empty front scenarios because statistics may be computed from fewer seeds than the reported successful run count.
