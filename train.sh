cd ../.. && source /app/bin/proxy.sh && cd ./HDD_POOL/ROLL
ssh -fN -L 18000:localhost:8000 -p 30115 root@121.46.19.2 
ssh -fN -L 18001:localhost:8000 -p 30116 root@121.46.19.2 
pip install tenacity

nohup python roll/pipeline/agentic/env/android/GuiTaskEvalManager.py \
    > roll/pipeline/agentic/env/android/GuiTaskEvalManager.log 2>&1 &
sh jyc/scripts/evaluate_agentic_pipeline.sh
sh jyc/scripts/run_agentic_pipeline.sh


#  Allocated group 0 for task: ExpenseDeleteSingle
#  Allocated group 1 for task: RecipeAddMultipleRecipes
#  Allocated group 2 for task: ClockStopWatchPausedVerify
#  Allocated group 3 for task: SimpleCalendarAddOneEventRelativeDay
#  Allocated group 4 for task: MarkorChangeNoteContent
#  Allocated group 5 for task: SystemWifiTurnOff
#  Allocated group 6 for task: MarkorCreateNoteAndSms
#  Allocated group 7 for task: RecipeAddMultipleRecipesFromMarkor2
#  Allocated group 8 for task: SimpleSmsSend


