from __future__ import annotations

import json
from pathlib import Path

TRAIN_TASK_LIST = [
    "SimpleSmsReply",
    "MarkorEditNote",
    "ExpenseDeleteMultiple2",
    "SystemWifiTurnOn",
    "FilesDeleteFile",
    "SystemBrightnessMin",
    "SimpleCalendarAddOneEventInTwoWeeks",
    "SystemBrightnessMax",
    "ClockTimerEntry",
    "SystemBluetoothTurnOn",
    "SimpleCalendarAddOneEventRelativeDay",
    "RecipeDeleteMultipleRecipesWithNoise",
    "MarkorDeleteAllNotes",
    "SimpleCalendarAddOneEventTomorrow",
    "SimpleSmsSendClipboardContent",
    "FilesMoveFile",
    "RecipeDeleteMultipleRecipesWithConstraint",
    "SimpleCalendarAddOneEvent",
    "CameraTakeVideo",
    "MarkorMoveNote",
    "SystemWifiTurnOff",
    "MarkorCreateNoteFromClipboard",
    "SaveCopyOfReceiptTaskEval",
    "RecipeDeleteDuplicateRecipes",
    "TurnOnWifiAndOpenApp",
    "ExpenseDeleteDuplicates2",
]

SUCCESS_TASK_LIST = [ # 成功率为100%的任务，共25个，这是对于基座模型而言的
    'AudioRecorderRecordAudio',
    'BrowserMaze',
    'ClockStopWatchPausedVerify',
    'ClockStopWatchRunning',
    'ContactsAddContact',
    'ContactsNewContactDraft',
    'ExpenseAddSingle',
    'ExpenseDeleteDuplicates',
    'ExpenseDeleteMultiple',
    'ExpenseDeleteSingle',
    'OpenAppTaskEval',
    'RecipeDeleteMultipleRecipes',
    'RecipeDeleteSingleRecipe',
    'RecipeDeleteSingleWithRecipeWithNoise',
    'SimpleCalendarAddRepeatingEvent',
    'SimpleCalendarDeleteEvents',
    'SimpleCalendarDeleteEventsOnRelativeDay',
    'SimpleCalendarDeleteOneEvent',
    'SimpleSmsSend',
    'SystemBluetoothTurnOffVerify',
    'SystemBluetoothTurnOnVerify',
    'SystemBrightnessMaxVerify',
    'SystemBrightnessMinVerify',
    'SystemWifiTurnOffVerify',
    'SystemWifiTurnOnVerify'
]



FAIL_TASK_LIST = [
    'AudioRecorderRecordAudioWithFileName',
    'BrowserDraw',
    'BrowserMultiply',
    'ExpenseAddMultiple',
    'ExpenseAddMultipleFromGallery',
    'ExpenseAddMultipleFromMarkor',
    'MarkorAddNoteHeader',
    'MarkorChangeNoteContent',
    'MarkorCreateFolder',
    'MarkorCreateNote',
    'MarkorCreateNoteAndSms',
    'MarkorDeleteNewestNote',
    'MarkorDeleteNote',
    'MarkorMergeNotes',
    'MarkorTranscribeReceipt',
    'MarkorTranscribeVideo',
    'OsmAndFavorite',
    'OsmAndMarker',
    'OsmAndTrack',
    'RecipeAddMultipleRecipes',
    'RecipeAddMultipleRecipesFromImage',
    'RecipeAddMultipleRecipesFromMarkor',
    'RecipeAddMultipleRecipesFromMarkor2',
    'RecipeAddSingleRecipe',
    'RecipeDeleteDuplicateRecipes2',
    'RecipeDeleteDuplicateRecipes3',
    'RetroCreatePlaylist',
    'RetroPlaylistDuration',
    'SimpleDrawProCreateDrawing',
    'SimpleSmsSendReceivedAddress',
    'SystemBluetoothTurnOff',
    'SystemCopyToClipboard',
    'TurnOffWifiAndTurnOnBluetooth',
    'VlcCreatePlaylist',
    'VlcCreateTwoPlaylists'
]

DEFAULT_TASK_SUBSETS: dict[str, list[str]] = {
    "全部任务": [],
    "TRAIN_TASK_LIST": TRAIN_TASK_LIST,
    "FAIL_TASK_LIST": FAIL_TASK_LIST,
    "SUCCESS_TASK_LIST": SUCCESS_TASK_LIST,
}


def load_task_subsets(config_path: Path) -> dict[str, list[str]]:
    subsets = dict(DEFAULT_TASK_SUBSETS)
    if not config_path.exists():
        return subsets

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            ext_data = json.load(f)
    except Exception:
        return subsets

    if not isinstance(ext_data, dict):
        return subsets

    for subset_name, tasks in ext_data.items():
        if not isinstance(subset_name, str):
            continue
        if not isinstance(tasks, list):
            continue
        cleaned_tasks = sorted({str(task) for task in tasks if str(task).strip()})
        subsets[subset_name] = cleaned_tasks

    return subsets
