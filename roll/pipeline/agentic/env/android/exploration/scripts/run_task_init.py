"""
CLI script for running task initialization exploration.

This script replaces the original Context Review with OpenMobile-style task initialization:
- Generates and persists deterministic params for each task
- Initializes tasks using reset_with_params endpoint
- Verifies initialization success with screenshots
- Saves task_init_result.json for each task

Usage:
    # Initialize all AndroidWorld tasks
    python run_task_init.py --env androidworld --server_url http://localhost:8000

    # Initialize specific tasks
    python run_task_init.py --task_pool ContactsAddContact,SimpleCalendarAddOneEvent

    # Initialize with multiple instances
    python run_task_init.py --num_instances 3 --output_dir ./init_output

    # MobileWorld initialization
    python run_task_init.py --env mobileworld --server_url http://localhost:18000
"""

import argparse
import json
import sys
from pathlib import Path

from roll.pipeline.agentic.env.android.exploration import (
    AndroidWorldTaskInitializer,
    MobileWorldTaskInitializer,
    AndroidWorldParamsManager,
    MobileWorldParamsManager,
)
from roll.pipeline.agentic.env.android.tasks import (
    TASK_LIST,
    TRAIN_TASK_LIST,
    Information_Retrieval_TASK_LIST,
    MOBILEWORLD_TASK_LIST,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run task initialization exploration for AndroidWorld or MobileWorld"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="androidworld",
        choices=["androidworld", "mobileworld"],
        help="Environment type",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8000",
        help="Server URL for environment API",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./androidworld_init_output",
        help="Output directory for initialization results",
    )
    parser.add_argument(
        "--params_dir",
        type=str,
        default=None,
        help="Directory for params pickle files (default: <output_dir>/params)",
    )
    parser.add_argument(
        "--task_pool",
        type=str,
        default=None,
        help="Comma-separated task names or 'auto' for all tasks",
    )
    parser.add_argument(
        "--task_family",
        type=str,
        default="android_world",
        help="Task family to use",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=1,
        help="Number of instances per task",
    )
    parser.add_argument(
        "--task_random_seed",
        type=int,
        default=42,
        help="Random seed for deterministic param generation",
    )
    parser.add_argument(
        "--max_init_steps",
        type=int,
        default=5,
        help="Maximum verification steps after initialization",
    )
    parser.add_argument(
        "--enable_vlm_verification",
        action="store_true",
        help="Enable VLM-based verification of initialization",
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
    return parser.parse_args()


def get_task_list(task_pool_arg: str, env: str) -> list:
    """Parse task pool argument into actual task list."""
    if task_pool_arg is None:
        if env == "androidworld":
            return TRAIN_TASK_LIST
        else:
            return MOBILEWORLD_TASK_LIST[:10]

    if task_pool_arg.lower() == "auto":
        if env == "androidworld":
            return TRAIN_TASK_LIST
        else:
            return MOBILEWORLD_TASK_LIST
    elif task_pool_arg.lower() == "all":
        if env == "androidworld":
            return TASK_LIST + Information_Retrieval_TASK_LIST
        else:
            return MOBILEWORLD_TASK_LIST
    elif task_pool_arg.lower() == "train":
        return TRAIN_TASK_LIST
    elif task_pool_arg.lower() == "info_retrieval":
        return Information_Retrieval_TASK_LIST
    else:
        return [t.strip() for t in task_pool_arg.split(",")]


def main():
    args = parse_args()

    task_list = get_task_list(args.task_pool, args.env)
    print(f"Task pool: {len(task_list)} tasks")

    params_dir = args.params_dir
    if params_dir is None:
        env_name = "androidworld" if args.env == "androidworld" else "mobileworld"
        params_dir = str(Path(args.output_dir) / "params" / env_name)

    if args.env == "androidworld":
        params_manager = AndroidWorldParamsManager(
            params_dir=params_dir,
            task_random_seed=args.task_random_seed,
        )
        initializer = AndroidWorldTaskInitializer(
            server_url=args.server_url,
            params_manager=params_manager,
            max_init_steps=args.max_init_steps,
            output_dir=args.output_dir,
            task_pool=task_list,
            enable_vlm_verification=args.enable_vlm_verification,
            console_port=args.console_port,
            grpc_port=args.grpc_port,
            adb_path=args.adb_path,
            task_family=args.task_family,
        )
    else:
        params_manager = MobileWorldParamsManager(
            params_dir=params_dir,
            task_random_seed=args.task_random_seed,
        )
        initializer = MobileWorldTaskInitializer(
            server_url=args.server_url,
            params_manager=params_manager,
            max_init_steps=args.max_init_steps,
            output_dir=args.output_dir,
            task_pool=task_list,
            enable_vlm_verification=args.enable_vlm_verification,
            device_id=args.device_id,
            snapshot=args.snapshot,
        )

    print(f"Environment: {args.env}")
    print(f"Server URL: {args.server_url}")
    print(f"Output directory: {args.output_dir}")
    print(f"Params directory: {params_dir}")
    print(f"Num instances per task: {args.num_instances}")
    print(f"Task random seed: {args.task_random_seed}")
    print()

    print("Starting task initialization...")
    results = initializer.run(num_instances=args.num_instances)

    success_count = sum(1 for r in results.values() if r.success)
    total_count = len(results)

    print()
    print("=" * 60)
    print("Task Initialization Complete!")
    print("=" * 60)
    print(f"Total tasks: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print()

    summary = {
        "env": args.env,
        "total_tasks": total_count,
        "successful": success_count,
        "failed": total_count - success_count,
        "results": {k: v.to_dict() for k, v in results.items()},
    }
    summary_path = Path(args.output_dir) / "initialization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {summary_path}")
    print()

    if total_count - success_count > 0:
        print("Failed tasks:")
        for key, result in results.items():
            if not result.success:
                print(f"  - {key}: {result.error_message or 'Unknown error'}")
        print()

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
