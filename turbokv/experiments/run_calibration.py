"""CLI runner for empirical calibration across configurations."""

import argparse
from turbokv.config.model_config import ModelConfig
from turbokv.config.policy_config import CacheConfig
from turbokv.inference.model_loader import ModelLoader
from turbokv.optimization.calibration import CalibrationRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Run TurboKV Calibration Suite")
    parser.add_argument("--model-path", type=str, default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--output-path", type=str, default="results/calibration_profile.json")
    return parser.parse_args()


def main():
    args = parse_args()
    model_cfg = ModelConfig(model_path=args.model_path)
    model, tokenizer, model_cfg = ModelLoader.load_model(model_cfg)

    configs = [
        CacheConfig("fp16"),
        CacheConfig("int8", group_size=64, quantized_start=0),
        CacheConfig("int8", group_size=32, quantized_start=0),
        CacheConfig("int8", group_size=64, quantized_start=512),
        CacheConfig("int4", group_size=64, quantized_start=0),
        CacheConfig("int4", group_size=32, quantized_start=0),
        CacheConfig("int4", group_size=64, quantized_start=512),
    ]

    runner = CalibrationRunner(model, tokenizer, model_config=model_cfg)
    runner.run_calibration(
        context_lengths=args.context_lengths,
        candidate_configs=configs,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
