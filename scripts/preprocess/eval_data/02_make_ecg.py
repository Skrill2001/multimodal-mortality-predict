import wfdb
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample_poly

STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
TARGET_FS = 500


def pad_to_12_leads(input_path, output_dir, record_name=None):
    record = wfdb.rdrecord(input_path)
    sig_name = record.sig_name
    orig_fs = record.fs
    duration = record.p_signal.shape[0]

    # 构造空的 (duration, 12) ECG 信号矩阵
    full_signal = np.zeros((duration, 12), dtype=np.float32)
    full_units = []

    sig_name_map = {name.upper(): idx for idx, name in enumerate(sig_name)}
    units_map = {name.upper(): record.units[idx] for idx, name in enumerate(sig_name)}

    for i, lead in enumerate(STANDARD_LEADS):
        lead_upper = lead.upper()
        if lead_upper in sig_name_map:
            idx = sig_name_map[lead_upper]
            full_signal[:, i] = record.p_signal[:, idx]
            full_units.append(units_map[lead_upper])
        else:
            # 缺失导联：填0，单位设默认
            full_units.append("mV")

    upsampled_signal = resample_poly(full_signal, up=TARGET_FS, down=orig_fs, axis=0)

    if record_name is None:
        record_name = os.path.basename(input_path)

    wfdb.wrsamp(
        record_name=record_name,
        fs=TARGET_FS,
        units=full_units,
        sig_name=STANDARD_LEADS,
        p_signal=upsampled_signal,
        fmt=['16'] * 12,
        write_dir=output_dir
    )


ORIG_ECG_DIR = "/Users/housiyuan/project/medical/ecg_segments/files"
STD_ECG_DIR = "/Users/housiyuan/project/medical/ecg_segments/files_std"
ecg_csv = "/Users/housiyuan/project/medical/ecg_segments/ecg_segment_summary.csv"

ecg_df = pd.read_csv(ecg_csv)
os.makedirs(STD_ECG_DIR)

for _, row in tqdm(ecg_df.iterrows(), total=len(ecg_df)):
    subject_id = int(row['subject_id'])
    input_path = os.path.join(ORIG_ECG_DIR, str(subject_id))
    output_path = STD_ECG_DIR
    pad_to_12_leads(input_path, output_path, record_name=str(subject_id))

