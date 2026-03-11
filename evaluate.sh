cd ../.. && source /app/bin/proxy.sh && ssh -fN -L 18000:localhost:8000 -p 30115 root@121.46.19.2 && cd ./HDD_POOL/ROLL
nohup python roll/pipeline/agentic/env/android/TaskEvalManager.py \
    > roll/pipeline/agentic/env/android/TaskEvalManager.log 2>&1 &
nohup python roll/pipeline/agentic/env/android/TaskManager.py \
    > roll/pipeline/agentic/env/android/TaskManager.log 2>&1 &
sh jyc/scripts/evaluate_agentic_pipeline.sh
sh jyc/scripts/run_agentic_pipeline.sh