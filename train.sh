cd ../.. && source /app/bin/proxy.sh && cd ./HDD_POOL/ROLL
ssh -fN -L 18000:localhost:8000 -p 30115 root@121.46.19.2 
ssh -fN -L 18001:localhost:8000 -p 30116 root@121.46.19.2 
pip install tenacity

nohup python roll/pipeline/agentic/env/android/GuiTaskEvalManager.py \
    > roll/pipeline/agentic/env/android/GuiTaskEvalManager.log 2>&1 &
sh jyc/scripts/evaluate_agentic_pipeline.sh
sh jyc/scripts/run_agentic_pipeline.sh




