"""
CLI script for running AndroidWorld/MobileWorld environment exploration.

Usage:
    # MobileWorld exploration with local vLLM model
    python run_exploration.py --env mobileworld \
        --server_url http://localhost:18000 \
        --model_backend vllm \
        --model_name Qwen2.5-VL-7B-Instruct \
        --vllm_base_url http://localhost:8000/v1

    # MobileWorld exploration with OpenAI API
    python run_exploration.py --env mobileworld \
        --server_url http://localhost:18000 \
        --model_backend openai \
        --model_name gpt-4o

    # Random exploration (no model)
    python run_exploration.py --env mobileworld --server_url http://localhost:18000

    # Custom settings
    python run_exploration.py --max_steps 100 --output_dir ./my_exploration_output
"""

import argparse
import sys
import uuid
from pathlib import Path

from roll.pipeline.agentic.env.android.exploration import (
    AndroidWorldExplorer,
    MobileWorldExplorer,
)
from roll.pipeline.agentic.env.android.exploration.model_client import (
    VLMModelFactory,
)


def _build_model_client(args: argparse.Namespace):
    """Build an ExplorerModelWrapper from CLI arguments."""
    if args.model_backend == "none" or args.model_backend is None:
        print("  Model: disabled (using random actions)")
        return None

    print(f"  Model backend: {args.model_backend}")
    print(f"  Model name:    {args.model_name}")
    print(f"  Base URL:      {getattr(args, 'vllm_base_url', 'default') or 'default'}")

    try:
        client = VLMModelFactory.create(
            backend=args.model_backend,
            model_name=args.model_name,
            base_url=getattr(args, "vllm_base_url", None),
            api_key=getattr(args, "api_key", None),
            temperature=args.model_temperature,
            max_tokens=args.model_max_tokens,
        )
        print(f"  Model client initialized: {args.model_name}")
        return client
    except Exception as e:
        print(f"  [WARNING] Failed to initialize model client: {e}")
        print("  Falling back to random actions.")
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run environment exploration for AndroidWorld or MobileWorld"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="androidworld",
        choices=["androidworld", "mobileworld"],
        help="Environment type to explore",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8000",
        help="Server URL for environment API",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum number of exploration steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./androidworld_exploration_output",
        help="Output directory for exploration results",
    )
    parser.add_argument(
        "--exploration_id",
        type=str,
        default=None,
        help="Custom exploration ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--console_port",
        type=int,
        default=5554,
        help="Android emulator console port (AndroidWorld only)",
    )
    parser.add_argument(
        "--grpc_port",
        type=int,
        default=8554,
        help="Android emulator gRPC port (AndroidWorld only)",
    )
    parser.add_argument(
        "--adb_path",
        type=str,
        default="/root/android-sdk/platform-tools/adb",
        help="Path to ADB executable",
    )
    parser.add_argument(
        "--device_id",
        type=str,
        default=None,
        help="Device ID (MobileWorld only, deprecated: use --console_port instead)",
    )
    parser.add_argument(
        "--console_port",
        type=int,
        default=5554,
        help="Android emulator console port (also used as device index for MobileWorld). Default: 5554",
    )
    parser.add_argument(
        "--grpc_port",
        type=int,
        default=8554,
        help="Android emulator gRPC port (also used as device index for MobileWorld). Default: 8554",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="default",
        help="Snapshot name. Default: default",
    )
    parser.add_argument(
        "--task_family",
        type=str,
        default="android_world",
        help="Task family for exploration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no_save_screenshots",
        action="store_true",
        help="Disable screenshot saving",
    )
    parser.add_argument(
        "--no_log_trajectory",
        action="store_true",
        help="Disable trajectory logging",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Custom exploration instruction",
    )

    # --- Model backend arguments ---
    model_group = parser.add_argument_group("VLM Model Configuration")
    model_group.add_argument(
        "--model_backend",
        type=str,
        default="none",
        choices=["none", "openai", "vllm", "huggingface"],
        help=(
            "VLM backend: 'none' (random actions), 'openai' (cloud/HF Inference API), "
            "'vllm' (local vLLM server), 'huggingface' (HF TGI server). "
            "Default: none"
        ),
    )
    model_group.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o",
        help="Model name. E.g. 'gpt-4o', 'Qwen/Qwen2.5-VL-7B-Instruct'",
    )
    model_group.add_argument(
        "--vllm_base_url",
        type=str,
        default=None,
        help=(
            "Base URL for vLLM / OpenAI-compatible endpoint. "
            "Example: 'http://localhost:8000/v1' "
            "(defaults to http://localhost:8000/v1 for vllm backend)"
        ),
    )
    model_group.add_argument(
        "--api_key",
        type=str,
        default=None,
        help=(
            "API key. For OpenAI cloud: reads OPENAI_API_KEY env var if not set. "
            "For local vLLM/HF TGI: defaults to 'EMPTY'"
        ),
    )
    model_group.add_argument(
        "--model_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for model generation. Default: 1.0",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of exploration episodes to run. Default: 1",
    )
    model_group.add_argument(
        "--model_max_tokens",
        type=int,
        default=256,
        help="Max tokens per model generation. Default: 256",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_client = _build_model_client(args)

    print(f"Starting {args.env} exploration...")
    print(f"  Exploration ID: {args.exploration_id}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Episodes: {args.num_episodes}")
    print()

    all_results = []
    for ep_idx in range(args.num_episodes):
        if args.num_episodes > 1:
            ep_exploration_id = f"{args.exploration_id}_ep{ep_idx:03d}" if args.exploration_id else None
            print(f"\n{'='*60}")
            print(f"  Episode {ep_idx + 1}/{args.num_episodes}")
            print(f"{'='*60}")
        else:
            ep_exploration_id = args.exploration_id

        if args.env == "androidworld":
            if "mobileworld" in args.output_dir.lower():
                ep_output_dir = "./androidworld_exploration_output"
            else:
                ep_output_dir = args.output_dir

            explorer = AndroidWorldExplorer(
                server_url=args.server_url,
                model_client=model_client,
                max_steps=args.max_steps,
                output_dir=ep_output_dir,
                exploration_id=ep_exploration_id,
                log_trajectory=not args.no_log_trajectory,
                save_screenshots=not args.no_save_screenshots,
                console_port=args.console_port,
                grpc_port=args.grpc_port,
                adb_path=args.adb_path,
                task_family=args.task_family,
                seed=args.seed + ep_idx,
                instruction=args.instruction,
            )
        else:
            if "androidworld" in args.output_dir.lower():
                ep_output_dir = "./mobileworld_exploration_output"
            else:
                ep_output_dir = args.output_dir

        explorer = MobileWorldExplorer(
            server_url=args.server_url,
            model_client=model_client,
            max_steps=args.max_steps,
            output_dir=ep_output_dir,
            exploration_id=ep_exploration_id,
            log_trajectory=not args.no_log_trajectory,
            save_screenshots=not args.no_save_screenshots,
            console_port=args.console_port,
            grpc_port=args.grpc_port,
            snapshot=args.snapshot,
            instruction=args.instruction,
        )

        result = explorer.run()
        explorer.save_trajectory()
        result_path = explorer.save_result()
        all_results.append(result)

    # Aggregate summary
    success_count = sum(1 for r in all_results if r.success)
    total_steps = sum(r.actual_steps for r in all_results)

    print()
    print("=" * 60)
    print("Exploration Complete!")
    print("=" * 60)
    print(f"  Total episodes: {len(all_results)}")
    print(f"  Successful: {success_count}/{len(all_results)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Model used: {all_results[0].model if all_results else 'N/A'}")
    print()

    if len(all_results) == 1:
        result = all_results[0]
        print(f"  Exploration ID: {result.exploration_id}")
        print(f"  Actual steps: {result.actual_steps}/{result.max_steps}")
        print(f"  Discovered apps: {', '.join(result.discovered_apps) if result.discovered_apps else 'None'}")
        print(f"  Discovered action types: {', '.join(result.discovered_action_types) if result.discovered_action_types else 'None'}")
        if result.error_message:
            print(f"  Error: {result.error_message}")
        print()
        print(f"  Result saved to: {result_path}")
        print(f"  Trajectory saved to: {result.trajectory_file}")
        if result.screenshots_dir:
            print(f"  Screenshots saved to: {result.screenshots_dir}")

    return 0 if success_count == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
