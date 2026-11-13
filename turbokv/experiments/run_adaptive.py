"""CLI runner for evaluating adaptive policy decisions across memory budgets and constraints."""

import argparse
import json
from turbokv.config.policy_config import PolicyConstraints, CacheConfig
from turbokv.optimization.adaptive_policy import AdaptiveKVPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TurboKV Adaptive Policy Decisions")
    parser.add_argument("--calibration-file", type=str, default="results/calibration_profile.json")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--memory-budgets-mb", type=float, nargs="+", default=[15.0, 30.0, 60.0, 100.0, 200.0])
    parser.add_argument("--max-acc-drop", type=float, default=0.05)
    parser.add_argument("--max-latency-regr-pct", type=float, default=20.0)
    parser.add_argument("--output-file", type=str, default="results/adaptive_decisions.json")
    return parser.parse_args()


def main():
    args = parse_args()
    policy = AdaptiveKVPolicy.from_calibration_file(args.calibration_file)

    all_decisions = []
    print("\n==================================================")
    print("TurboKV Adaptive Policy Simulation Matrix")
    print(f"Max Accuracy Drop Allowed: {args.max_acc_drop}")
    print(f"Max Latency Regression Allowed: {args.max_latency_regr_pct}%")
    print("==================================================")

    for budget in args.memory_budgets_mb:
        print(f"\n--- Budget: {budget:.1f} MB ---")
        constraints = PolicyConstraints(
            max_memory_budget_mb=budget,
            max_accuracy_drop=args.max_acc_drop,
            max_latency_regression_pct=args.max_latency_regr_pct,
        )
        for ctx in args.context_lengths:
            dec = policy.select(context_length=ctx, constraints=constraints)
            all_decisions.append(dec.to_dict())
            chosen = dec.chosen_config.key if dec.chosen_config else "NONE"
            print(f"  Ctx: {ctx:5d} -> Chosen: {chosen:15s} | Feasible: {dec.is_feasible!s:5s} | Reason: {dec.decision_reason}")

    with open(args.output_file, "w") as f:
        json.dump(all_decisions, f, indent=2)
    print(f"\nSaved decisions matrix to {args.output_file}")


if __name__ == "__main__":
    main()
