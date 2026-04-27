"""
CLI script for running AndroidWorld/MobileWorld environment exploration.

Usage:
    # AndroidWorld exploration with default settings
    python run_exploration.py --server_url http://localhost:8000

    # MobileWorld exploration
    python run_exploration.py --env mobileworld --server_url http://localhost:9000

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
        default="device_001",
        help="Device ID (MobileWorld only)",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="default",
        help="Snapshot name (MobileWorld only)",
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
        "--model_type",
        type=str,
        default="gui_owl",
        help="Model type for action generation",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Custom exploration instruction",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.env == "androidworld":
        if "mobileworld" in args.output_dir.lower():
            args.output_dir = "./androidworld_exploration_output"

        explorer = AndroidWorldExplorer(
            server_url=args.server_url,
            model_client=None,
            model_type=args.model_type,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            exploration_id=args.exploration_id,
            log_trajectory=not args.no_log_trajectory,
            save_screenshots=not args.no_save_screenshots,
            console_port=args.console_port,
            grpc_port=args.grpc_port,
            adb_path=args.adb_path,
            task_family=args.task_family,
            seed=args.seed,
            instruction=args.instruction,
        )
    else:
        if "androidworld" in args.output_dir.lower():
            args.output_dir = "./mobileworld_exploration_output"

        explorer = MobileWorldExplorer(
            server_url=args.server_url,
            model_client=None,
            model_type=args.model_type,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            exploration_id=args.exploration_id,
            log_trajectory=not args.no_log_trajectory,
            save_screenshots=not args.no_save_screenshots,
            device_id=args.device_id,
            snapshot=args.snapshot,
            instruction=args.instruction,
        )

    print(f"Starting {args.env} exploration...")
    print(f"  Exploration ID: {explorer.exploration_id}")
    print(f"  Output directory: {explorer.output_dir}")
    print(f"  Max steps: {args.max_steps}")
    print()

    result = explorer.run()
    explorer.save_trajectory()
    result_path = explorer.save_result()

    print()
    print("=" * 60)
    print("Exploration Complete!")
    print("=" * 60)
    print(f"  Exploration ID: {result.exploration_id}")
    print(f"  Actual steps: {result.actual_steps}/{result.max_steps}")
    print(f"  Discovered apps: {', '.join(result.discovered_apps) if result.discovered_apps else 'None'}")
    print(f"  Discovered action types: {', '.join(result.discovered_action_types) if result.discovered_action_types else 'None'}")
    print(f"  Success: {result.success}")
    if result.error_message:
        print(f"  Error: {result.error_message}")
    print()
    print(f"  Result saved to: {result_path}")
    print(f"  Trajectory saved to: {result.trajectory_file}")
    if result.screenshots_dir:
        print(f"  Screenshots saved to: {result.screenshots_dir}")
    print()

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
