def _evaluate_with_cuajudge(
    self,
    text_actions: List[str],
    next_observations: Dict,
    infos: List[Dict],
    dones: np.ndarray,
    original_rewards: np.ndarray,
    done_indices: List[int] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Evaluate rewards using CUAJudge for environments that are done.
    """

    # Convert to numpy if needed
    dones = to_numpy(dones)
    original_rewards = to_numpy(original_rewards)

    # Start with original rewards and deep copy infos to avoid modifying original data
    rewards = copy.deepcopy(
        original_rewards
    )  # Deep copy rewards to avoid modifying original
    updated_infos = copy.deepcopy(infos)

    # Use provided done_indices or find all done environments
    if done_indices is None:
        done_indices = np.where(dones)[0]

    if len(done_indices) == 0:
        print("[CUAJudge] No environments are done, using original rewards")
        return rewards, updated_infos

    print(f"[CUAJudge] Evaluating {len(done_indices)} done environments with CUAJudge")

    # Create temporary directory for saving images
    os.makedirs(self.cuajudge_temp_dir, exist_ok=True)
    print(f"[CUAJudge] Created/verified temp directory: {self.cuajudge_temp_dir}")

    # Create a unique subdirectory for this evaluation session
    # Use done_indices in the name to avoid conflicts when multiple futures run in parallel
    done_indices_str = "_".join(map(str, done_indices))
    temp_dir = os.path.join(
        self.cuajudge_temp_dir,
        f"cuajudge_session_{done_indices_str}_{int(time.time())}",
    )
    os.makedirs(temp_dir, exist_ok=True)

    # Create tasks for CUAJudge evaluation
    cuajudge_tasks = []
    for i in done_indices:
        # Get action history from buffer
        action_history = []
        if i < len(self.buffers) and self.buffers[i]:
            action_history = [rec["action"] for rec in self.buffers[i]]
        # Add current action
        if text_actions[i]:
            action_history.append(text_actions[i])

        # Save only historical images from buffer (in chronological order)
        all_image_paths = []

        if i < len(self.buffers) and self.buffers[i]:
            for j, record in enumerate(self.buffers[i]):
                if "image" in record and record["image"] is not None:
                    hist_image_path = os.path.join(temp_dir, f"task_{i}_hist_{j}.png")
                    hist_saved = self._save_image_for_cuajudge(
                        record["image"], hist_image_path
                    )
                    if hist_saved and os.path.exists(hist_image_path):
                        all_image_paths.append(hist_image_path)
                        print(
                            f"[CUAJudge] Saved historical image {j} for task {i}: {hist_image_path}"
                        )

            if all_image_paths:  # Only add task if we have images
                print(
                    f"[CUAJudge] Task {i} prepared with {len(all_image_paths)} historical images: {all_image_paths}"
                )

                task = {
                    "task": self.tasks[i],
                    "action_thoughts": None,
                    "last_actions": action_history,
                    "images_path": all_image_paths,
                    "input_image_paths": None,
                }
                cuajudge_tasks.append((i, task))
            else:
                print(
                    f"[CUAJudge] Warning: No historical images available for task {i}"
                )
        else:
            print(f"[CUAJudge] Warning: No buffer history available for task {i}")

    if not cuajudge_tasks:
        print("[CUAJudge] No valid tasks for CUAJudge evaluation")
        return rewards, updated_infos

    # Run CUAJudge evaluation asynchronously for all done tasks
    # Create event loop if not exists
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Create async tasks for parallel evaluation
    async def evaluate_single_task(task_idx, task_data):
        """Evaluate a single task asynchronously"""
        try:
            print(f"[CUAJudge] Starting evaluation for task {task_idx}...")

            # Verify all image files exist before calling CUAJudge
            image_paths = task_data["images_path"] if task_data["images_path"] else []
            missing_files = []
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    missing_files.append(img_path)

            if missing_files:
                raise FileNotFoundError(
                    f"Missing image files for task {task_idx}: {missing_files}"
                )

            print(
                f"[CUAJudge] Verified {len(image_paths)} image files exist for task {task_idx}"
            )

            # Run CUAJudge evaluation
            result = await self.cuajudge_evaluator.cuajudge_general_eval(
                task=task_data["task"],
                input_image_paths=task_data["input_image_paths"],
                action_thoughts=task_data["action_thoughts"],
                last_actions=task_data["last_actions"],
                images_path=task_data["images_path"],
            )

            # Unpack the result (must be a 3-tuple)
            if len(result) != 3:
                raise ValueError(
                    f"cuajudge_general_eval must return (response, reward, details); got: {result}"
                )
            response, reward, cuajudge_details = result

            print(f"[CUAJudge] Task {task_idx} evaluation completed: reward={reward}")

            return {
                "idx": task_idx,
                "response": response,
                "reward": reward,
                "cuajudge_details": cuajudge_details,
                "task_data": task_data,
                "success": True,
            }

        except Exception as e:
            print(f"[CUAJudge] Error evaluating task {task_idx}: {e}")
            return {"idx": task_idx, "error": str(e), "success": False}

    # Create all async tasks
    async_tasks = [
        evaluate_single_task(idx, task_data) for idx, task_data in cuajudge_tasks
    ]

    print(f"[CUAJudge] Starting parallel evaluation of {len(async_tasks)} tasks...")

    # Execute all tasks concurrently and wait for completion
    results = loop.run_until_complete(
        asyncio.gather(*async_tasks, return_exceptions=True)
    )

    print(f"[CUAJudge] All {len(results)} tasks completed")

    # Process results in order
    for result in results:
        if isinstance(result, Exception):
            print(f"[CUAJudge] Task evaluation failed with exception: {result}")
            continue

        if not result["success"]:
            # Handle failed evaluation
            idx = result["idx"]
            if idx < len(updated_infos):
                updated_infos[idx]["cuajudge_evaluated"] = False
                updated_infos[idx]["cuajudge_error"] = result["error"]
            continue

        # Process successful evaluation
        idx = result["idx"]
        response = result["response"]
        reward = result["reward"]
        cuajudge_details = result["cuajudge_details"]
        task_data = result["task_data"]

        # Update rewards and infos
        rewards[idx] = reward

        if idx < len(updated_infos):
            updated_infos[idx]["task_score"] = reward
            updated_infos[idx]["won"] = reward == 1.0
            updated_infos[idx]["reward"] = reward
            updated_infos[idx]["cuajudge_evaluated"] = True
            updated_infos[idx]["cuajudge_response"] = response

            # Add comprehensive CUAJudge evaluation details to info
            updated_infos[idx]["cuajudge_evaluation"] = {
                "response": response,
                "task_score": reward,
                "reward": reward,
                "predicted_label": int(reward == 1.0),
                "thoughts": cuajudge_details.get("thoughts", []),
                "image_judge_record": cuajudge_details.get("image_judge_record", []),
                "key_points": cuajudge_details.get("key_points", ""),
                "cuajudge_config": {
                    "key_model": getattr(
                        self.cuajudge_evaluator,
                        "key_identification_screenshot_model",
                        "unknown",
                    ),
                    "outcome_model": getattr(
                        self.cuajudge_evaluator, "key_points_outcome_model", "unknown"
                    ),
                    "max_image": getattr(self.cuajudge_evaluator, "max_image", 50),
                    "score_threshold": getattr(
                        self.cuajudge_evaluator, "score_threshold", 3
                    ),
                },
            }

            # Add rule-based comparison results only when explicitly enabled
            if self.enable_rule_based:
                original_info = infos[idx] if idx < len(infos) else {}
                if "rule_based_task_score" in original_info:
                    updated_infos[idx]["rule_based_comparison"] = {
                        "task_score": original_info.get("rule_based_task_score", 0.0),
                        "won": original_info.get("rule_based_won", False),
                        "reward": original_info.get("rule_based_reward", 0.0),
                        "evaluation_failed": original_info.get(
                            "rule_based_evaluation_failed", False
                        ),
                        "evaluation_error": original_info.get(
                            "rule_based_evaluation_error", None
                        ),
                        "evaluation_method": original_info.get(
                            "rule_based_evaluation_method", "rule_based"
                        ),
                    }
                else:
                    updated_infos[idx]["rule_based_comparison"] = {
                        "task_score": 0.0,
                        "won": False,
                        "reward": 0.0,
                        "evaluation_failed": True,
                        "evaluation_error": "Rule-based evaluation was enabled but no rule-based results were found",
                        "evaluation_method": "rule_based_missing",
                    }

            # Add task execution details
            updated_infos[idx]["task_execution"] = {
                "images_used": task_data["images_path"],
                "total_images": len(task_data["images_path"]),
                "evaluation_timestamp": datetime.now().isoformat(),
            }

        print(
            f"[CUAJudge] Task {idx} processed: reward={reward}, response={response[:100]}..."
        )

    # Safe cleanup: ignore errors if directory doesn't exist or was already deleted
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"[CUAJudge] Cleaned up temporary directory: {temp_dir}")
        else:
            print(
                f"[CUAJudge] Temporary directory already cleaned or doesn't exist: {temp_dir}"
            )
    except Exception as e:
        print(
            f"[CUAJudge] Warning: Failed to clean up temporary directory {temp_dir}: {e}"
        )
        # Continue execution - cleanup failure should not stop training

    return rewards, updated_infos


def save_cuajudge_results(
    self,
    task_name: str,
    task_id: str,
    cuajudge_results: Dict,
    save_dir: str = "cuajudge_results",
):
    """
    Save CUAJudge evaluation results to JSON file

    Args:
        task_name: Name of the task
        task_id: Unique identifier for the task
        cuajudge_results: Dictionary containing CUAJudge evaluation results
        save_dir: Directory to save the results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate filename with timestamp and task info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"CUAJudge_eval_{task_name}_{task_id}_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)

    # Add metadata to results
    cuajudge_results["metadata"] = {
        "task_name": task_name,
        "task_id": task_id,
        "timestamp": timestamp,
        "evaluation_method": "cuajudge_with_rule_based_comparison",
        "model_config": {
            "key_model": getattr(
                self.cuajudge_evaluator,
                "key_identification_screenshot_model",
                "unknown",
            ),
            "outcome_model": getattr(
                self.cuajudge_evaluator, "key_points_outcome_model", "unknown"
            ),
            "max_image": getattr(self.cuajudge_evaluator, "max_image", 50),
            "score_threshold": getattr(self.cuajudge_evaluator, "score_threshold", 3),
        },
        "note": "This file contains both CUAJudge and rule-based evaluation results for comparison",
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cuajudge_results, f, indent=2, ensure_ascii=False)
    print(f"[CUAJudge] Results saved to {filepath}")
    return filepath
