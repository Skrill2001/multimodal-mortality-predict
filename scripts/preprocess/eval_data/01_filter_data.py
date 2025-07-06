import os
import pandas as pd
import wfdb
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np

PATIENT_CSV = "/Users/housiyuan/project/medical/bypy/data_all.csv"
# MIMIC_ROOT = '/Users/housiyuan/project/medical/bypy'
MIMIC_ROOT = '/Volumes/EXTERNAL_USB/mimic-新版cx'
OUTPUT_DIR = '/Users/housiyuan/project/medical/ecg_segments'
SUMMARY_CSV = os.path.join(OUTPUT_DIR, 'ecg_segment_summary.csv')
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "files")

MIN_REQUIRED_LEADS = 1  # 最小导联数要求
SEGMENT_DURATION_SEC = 10
OFFSET_HOURS = 1  # 从数据开始后的1小时处截取
MAX_ATTEMPT = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)
summary_rows = []

# === 读取患者列表 ===
patient_df = pd.read_csv(PATIENT_CSV)
print(f"共读取 {len(patient_df)} 位患者")

# === 遍历患者 ===
for _, row in tqdm(patient_df.iterrows(), total=len(patient_df)):
    subject_id = int(row['SUBJECT_ID'])
    subj_folder = f"p{str(subject_id).zfill(6)[:2]}/p{str(subject_id).zfill(6)}"
    subj_path = os.path.join(MIMIC_ROOT, subj_folder)

    if not os.path.isdir(subj_path):
        continue

    # === 找主头文件：格式为 pxxxxxx-YYYY-MM-DD-HH-MM.hea
    main_header_files = [f for f in os.listdir(subj_path) if f.endswith('.hea') and '-' in f and f.startswith(f"p{str(subject_id).zfill(6)}")]
    if not main_header_files:
        print(f"[Waring] Cannot find main header file in {subj_path}")
        continue

    candidates = []

    for main_header_file in main_header_files:
        main_header_path = os.path.join(subj_path, main_header_file)
        header_type = os.path.basename(main_header_path).split('.')[0][-1]

        # 排除心率等数字记录
        if header_type.isalpha():
            continue

        with open(main_header_path, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2:
            continue

        # 解析主记录起始日期（第一行）
        base_info = lines[0].strip().split()
        try:
            base_date_str = base_info[5]
            main_date = datetime.strptime(base_date_str, "%d/%m/%Y").date()
        except:
            main_date = None

        for line in lines[1:]:
            record_id = line.strip().split()[0]

            if line.startswith('~') or ('layout' in line) or (len(line.strip().split()) < 2) or (record_id[-1].isalpha()):
                continue

            rec_path = os.path.join(subj_path, record_id)
            if not os.path.exists(rec_path + '.hea'):
                continue

            try:
                header = wfdb.rdheader(rec_path)
                fs = header.fs
                sig_len = header.sig_len
                n_sig = header.n_sig
                sub_time = header.base_time

                if n_sig < MIN_REQUIRED_LEADS:
                    continue

                segment_len = int(fs * SEGMENT_DURATION_SEC)
                offset_start = int(fs * OFFSET_HOURS * 3600)
                offset_step = int(fs * 600)  # 每次顺延 10 分钟

                for attempt in range(MAX_ATTEMPT):
                    offset_samples = offset_start + attempt * offset_step
                    if offset_samples + segment_len > sig_len:
                        break

                    record = wfdb.rdrecord(rec_path,
                                           sampfrom=offset_samples,
                                           sampto=offset_samples + segment_len)
                    p = record.p_signal

                    # 检查无效信号（NaN 或 全0）
                    if np.isnan(p).any() or (p.std(axis=0) == 0).any():
                        continue

                    # 检查重复导联
                    if len(record.sig_name) != len(set(record.sig_name)):
                        continue

                    # 有效片段，准备保存候选
                    if main_date is not None and sub_time is not None:
                        sub_start_time = datetime.combine(main_date, sub_time)
                        segment_time = sub_start_time + timedelta(seconds=offset_samples / fs)
                    else:
                        segment_time = datetime.max

                    candidates.append({
                        'record_id': record_id,
                        'index': int(record_id.split('_')[-1]),
                        'fs': fs,
                        'n_sig': n_sig,
                        'segment_time': segment_time,
                        'record': record,
                        'record_header': header,
                        'main_header': main_header_file
                    })

                    break

            except Exception as e:
                print(f"跳过 {record_id}: {e}")
                continue

    # === 选择最佳候选 ===
    if len(candidates) > 0:
        candidates_sorted = sorted(
            candidates,
            key=lambda x: (-x['n_sig'], -x['fs'], x['index'], x['segment_time'])
        )
        best = candidates_sorted[0]

        # === 保存为 .hea/.dat ===
        wfdb.wrsamp(record_name=str(subject_id),
                    fs=best['fs'],
                    units=best['record'].units,
                    sig_name=best['record'].sig_name,
                    p_signal=best['record'].p_signal,
                    fmt=['16'] * best['n_sig'],
                    write_dir=OUTPUT_DIR)

        # === 保存 summary 记录 ===
        summary_rows.append({
            'subject_id': subject_id,
            'main_header': best['main_header'],
            'record': best['record_id'],
            'ecg_time': best['segment_time'].strftime("%Y-%m-%d %H:%M:%S"),
            'sig_name': best['record'].sig_name,
            'fs': best['fs']
        })
    else:
        print(f"[Warning] Cannot find candidate file in {subj_path}")

# === 保存summary.csv ===
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"已保存summary至 {SUMMARY_CSV}, 总共处理成功 {len(summary_rows)}")
