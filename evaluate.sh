cd ../.. && source /app/bin/proxy.sh && cd ./HDD_POOL/ROLL
ssh -fN -L 18000:localhost:8000 -p 30115 root@121.46.19.2 
ssh -fN -L 18001:localhost:8000 -p 30116 root@121.46.19.2 
pip install tenacity

nohup python roll/pipeline/agentic/env/android/GuiTaskEvalManager.py \
    > roll/pipeline/agentic/env/android/GuiTaskEvalManager.log 2>&1 &
sh jyc/scripts/evaluate_agentic_pipeline.sh reflection

# ========================================================================
cd ../.. && source /app/bin/proxy.sh && cd ./HDD_POOL/ROLL
ssh -fN -L 18000:localhost:8000 -p 30115 root@121.46.19.2 
ssh -fN -L 18001:localhost:8000 -p 30116 root@121.46.19.2 
pip install tenacity

nohup python roll/pipeline/agentic/env/android/GuiTaskEvalManager.py \
    > roll/pipeline/agentic/env/android/GuiTaskEvalManager.log 2>&1 &
sh jyc/scripts/run_agentic_pipeline.sh

# cd ~/HDD_POOL/ROLL/roll/pipeline/agentic/env/android && conda activate roll && streamlit run app.py
# ssh -L 8501:localhost:8501 超算

#  Allocated group 0 for task: ExpenseDeleteSingle
#  Allocated group 1 for task: RecipeAddMultipleRecipes
#  Allocated group 2 for task: ClockStopWatchPausedVerify
#  Allocated group 3 for task: SimpleCalendarAddOneEventRelativeDay
#  Allocated group 4 for task: MarkorChangeNoteContent
#  Allocated group 5 for task: SystemWifiTurnOff
#  Allocated group 6 for task: MarkorCreateNoteAndSms
#  Allocated group 7 for task: RecipeAddMultipleRecipesFromMarkor2
#  Allocated group 8 for task: SimpleSmsSend

evaluate：
PID     MEM(GB) COMMAND
2723    35.02   ray::RolloutScheduler.get_batch
2861    31.17   ray::GroupQueueManager
2915    22.56   ray::EnvironmentWorker
2833    6.96    VLLM::EngineCore
2498    1.01    ray::ActorWorker
2860    0.78    ray::RequestScheduler.generate_one_request
296     0.59    python jyc/evaluate_agentic_pipeline.py --config_path . --config_name agent_val_multiandroid_grpo_ev...
1523    0.09    /usr/bin/python -u /usr/local/lib/python3.12/dist-packages/ray/dashboard/agent.py --node-ip-address=...
652     0.08    /usr/bin/python /usr/local/lib/python3.12/dist-packages/ray/dashboard/dashboard.py --host=127.0.0.1 ...
784     0.08    ray-dashboard-ReportHead-0 (/usr/bin/python -c "from multiprocessing.spawn import spawn_main; spawn_...


