cd ../.. && source /app/bin/proxy.sh && cd ./HDD_POOL/ROLL
ssh -fN -L 18000:localhost:8000 -p 30115 root@121.46.19.2 
pip install tenacity
# ssh -fN -L 18001:localhost:8000 -p jiang_yicheng@10.249.43.2
nohup python roll/pipeline/agentic/env/android/GuiTaskEvalManager.py \
    > roll/pipeline/agentic/env/android/GuiTaskEvalManager.log 2>&1 &
sh jyc/scripts/evaluate_agentic_pipeline.sh

sh jyc/scripts/run_agentic_pipeline.sh


