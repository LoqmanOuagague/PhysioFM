# Model backbones supported across the downstream/zero-shot pipelines.
MODEL_LIST = ['ast', 'clap', 'opera', 'normwear', 'chronos']

# Per-dataset task specification, keyed by "<domain>/<dataset_name>" matching the
# folder under NormWear/data/. Each dataset can bundle multiple sub-tasks
# (e.g. multiple labels predicted from the same embedding), so every value below
# is a list indexed by task position (task_idx), aligned with the dataset's
# label order.
#
# Keys per entry:
#   "nums":   for a classification task ('class'), the number of classes;
#             for a regression task ('reg'), the output dimensionality
#             (1 for scalar regression, >1 for multi-output regression).
#   "names":  display/log name for each task, used to key the results dict
#             (task_scores) and to label task outputs in zero-shot inference.
#   "ranges": optional, regression tasks only. [low, high] bounds of the target
#             value, used to build a SigmoidRange output head (NormWear/modules/head.py)
#             that squashes predictions into that range. `None` at a given
#             position means no bounding for that task. Omitted entirely for
#             datasets with no regression tasks.
CLASS_NUM = {
    "audio_downstream/KAUH": {
        "nums": [8],
        "names": ["single task"]
    },
    "audio_downstream/RespiratoryDatabase@TR": {
        "nums": [5],
        "names": ["single task"]
    },
    "audio_downstream/FSD50K": {
        "nums": [2 for _ in range(5)],
        "names": ["breathing", "cough", "laughter", "sneeze", "speech"]
    },
    "audio_downstream/NoseMic": {
        "nums": [2, 1],
        "ranges": [None, [1, 60]],
        "names": ["run_state", "resp_rate"]
    },
    "audio_downstream/Coswara": {
        "nums": [2 for _ in range(16)],
        "names": [
            "smoker",
            "cold",
            "ht",
            "diabetes",
            "cough",
            "diarrhoea",
            "fever",
            "loss_of_smell",
            "bd",
            "st",
            "ihd",
            "asthma",
            "cld",
            "pneumonia",
            "ftg",
            "mp",
        ]
    },
    "audio_downstream/FluSense": {
        "nums": [2 for _ in range(7)],
        "names": [
            "sneeze",
            "sniffle",
            "cough",
            "gasp",
            "breathe",
            "speech",
            "throat-clearing"
        ]
    },
    # ===== WEARABLE STARTING HERE ============================
    "wearable_downstream/PPG_HTN": {
        "nums": [4],
        "names": ["PPG_HTN"]
    },
    "wearable_downstream/PPG_DM": {
        "nums": [2],
        "names": ["PPG_DM"]
    },
    "wearable_downstream/PPG_CVA": {
        "nums": [2],
        "names": ["PPG_CVA"]
    },
    "wearable_downstream/PPG_CVD": {
        "nums": [3],
        "names": ["PPG_CVD"]
    },
    "wearable_downstream/indian-fPCG": {
        "nums": [1],
        "ranges": [[60, 200]], 
        "names": ["indian-fPCG"]
    },
    "wearable_downstream/ppg_hgb": {
        "nums": [1],
        "ranges": [[1, 20]],
        "names": ["ppg_hgb"]
    },
    "wearable_downstream/non_invasive_bp": {
        "nums": [2],
        "ranges": [[-4, 3]],
        "names": ["non_invasive_bp"]
    },
    "wearable_downstream/drive_fatigue": {
        "nums": [2],
        "names": ["drive_fatigue"]
    },
    "wearable_downstream/ecg_heart_cat": {
        "nums": [2],
        "names": ["ecg_heart_cat"]
    },
    "wearable_downstream/gameemo": {
        "nums": [4],
        "names": ["gameemo"]
    },
    "wearable_downstream/uci_har": {
        "nums": [6],
        "names": ["uci_har"]
    },
    "wearable_downstream/wesad": {
        "nums": [3],
        "names": ["wesad"]
    },
    "wearable_downstream/Cogload": {
        "nums": [6],
        "ranges": [[1, 100] for _ in range(6)],
        "names": ["Cogload"]
    },
    "wearable_downstream/Epilepsy": {
        "nums": [2 for _ in range(5)],
        "names": [
            "A_Z_eye_open",
            "B_O_eye_close",
            "C_N_health",
            "D_F_tumor",
            "E_S_seizure"
        ]
    },
    "wearable_downstream/emg-tfc": {
        "nums": [3],
        "names": ["emg-tfc"]
    },
    "wearable_downstream/ecg-tfc": {
        "nums": [4],
        "names": ["ecg-tfc"]
    },
    # ===== CLINIC STARTING HERE ============================
    "clinic_data/swallow_data": {
        "nums": [3],
        "names": ["single task"]
    },
    "clinic_data/studentlife": { # every 6 hours as a unit, i.e. sr=6 (1 data point is of 1 hour)
        "nums": [3],
        "names": ["single task"]
    },
    "clinic_data/opioid_misuse": { # every 15 mins as an unit, i.e. sr=6 (1 data point is of 2.5 mins)
        "nums": [1, 1, 1, 2],
        "ranges": [[1, 11] for _ in range(3)],
        "names": [
            "stress", 
            "pain",
            "craving",
            "misuse_risk"
        ]
    },
    "clinic_data/IDH_risk": { # every 1 hour as a unit, i.e. sr=12 (1 data point is of 5 mins)
        "nums": [2 for _ in range(5)],
        "names": [
            "fall20", 
            "fall30",
            "nadir90",
            "nadir100",
            "hemo"
        ]
    },
    # ===== RADAR STARTING HERE ============================
    "radar_data/radar_9_subjects": { # every 1 hour as a unit, i.e. sr=12 (1 data point is of 5 mins)
        "nums": [1, 1, 2],
        "names": [
            "hr", 
            "resp_r",
            "resp_abnormal"
        ]
    },
    "radar_data/radar_30_subjects": { # every 1 hour as a unit, i.e. sr=12 (1 data point is of 5 mins)
        "nums": [5, 1, 1, 4, 1, 2],
        "names": [
            "action", 
            "sysbp",
            "diabp",
            "bp_abnormal",
            "hr",
            "hr_abnormal"
        ]
    },
}