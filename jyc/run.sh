# source /APP/u22/ai_x86/toolshs/setproxy.sh 172.16.37.200 3138
# source /APP/u22/ai_x86/toolshs/network.sh hitsz_xdeng_2 Db7AQXLzNov04S7i

pip install tenacity
export ROLLOUT_CACHE_MONITOR_INTERVAL=10
export ROLLOUT_OBJ_PROBE_ENABLE=1
export ROLLOUT_OBJ_PROBE_EVERY_N=3

cd ../.. && source /app/bin/proxy.sh && cd ./HDD_POOL/ROLL
ssh -fN -L 18000:localhost:8000 -p 30115 root@121.46.19.2 
ssh -fN -L 18001:localhost:8000 -p 30116 root@121.46.19.2 

bash jyc/scripts/start_task_manager.sh both
sh jyc/scripts/evaluate_agentic_pipeline.sh voyager mobile_world
sh jyc/scripts/evaluate_agentic_pipeline.sh qwen mobile_world


# nohup python roll/pipeline/agentic/env/android/GuiTaskEvalManager.py \
#     > roll/pipeline/agentic/env/android/GuiTaskEvalManager.log 2>&1 &


# ========================================================================
pip install tenacity
pip install nvidia-ml-py

export RAY_DEBUG=legacy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ../.. && source /app/bin/proxy.sh && cd ./HDD_POOL/ROLL
ssh -fN -L 18000:localhost:8000 -p 30115 root@121.46.19.2 
ssh -fN -L 18001:localhost:8000 -p 30116 root@121.46.19.2 

nohup python -u roll/pipeline/agentic/env/android/gpu_monitor.py \
    > roll/pipeline/agentic/env/android/monitor.log 2>&1 &

bash jyc/scripts/start_task_manager.sh both
sh jyc/scripts/run_agentic_pipeline.sh


# cd ~/HDD_POOL/ROLL/roll/pipeline/agentic/env/android && conda activate roll && streamlit run app_modular.py
# ssh -L 8501:localhost:8501 超算


# ========================================================================
apt update && apt install -y openssh-server
/app/bin/sshdctl start

source /app/bin/proxy.sh
/app/ubuntu/bin/apt-get update
/app/ubuntu/bin/apt-get install tmux
