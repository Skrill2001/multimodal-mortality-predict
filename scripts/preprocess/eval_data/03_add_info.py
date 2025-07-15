import pandas as pd
from datetime import datetime, timedelta
import wfdb
import os
from tqdm import tqdm


def corrected_ecg_time(row, ecg_root_dir):
    subject_id = int(row['subject_id'])
    main_header = row['main_header']
    record_name = row['record']
    ecg_time = pd.to_datetime(row['ecg_time'])

    subj_path = os.path.join(ecg_root_dir, f"p{str(subject_id).zfill(6)[:2]}/p{str(subject_id).zfill(6)}")
    main_header_path = os.path.join(subj_path, main_header)
    record_path = os.path.join(subj_path, record_name)

    try:
        # 读取 main_header 时间
        with open(main_header_path, 'r') as f:
            parts = f.readline().strip().split()
        main_date = datetime.strptime(parts[5], "%d/%m/%Y").date()
        main_time = datetime.strptime(parts[4], "%H:%M:%S.%f").time()
        main_datetime = datetime.combine(main_date, main_time)
    except Exception as e:
        print(f"\n[WARNING]: GET MEAN HEADER TIME FAILED, HEADER PATH: {main_header_path}, ERROR: {e}")
        return ecg_time  # 保守处理，返回原始时间

    try:
        # 读取 record base_time
        record_header = wfdb.rdheader(record_path)
        base_time = record_header.base_time
        if base_time is None:
            print(f"[WARNING]: GET RECORD TIME FAILED, RECORD PATH: {record_path}")
            return ecg_time
        record_datetime = datetime.combine(main_date, base_time)
    except:
        print(f"[WARNING]: GET RECORD TIME FAILED, RECORD PATH: {record_path}")
        return ecg_time

    # 判断是否跨天
    if record_datetime < main_datetime:
        return ecg_time + timedelta(days=1)
    else:
        return ecg_time


# 路径自行修改
base_csv_path = "/Users/housiyuan/project/medical/ecg_segments/ecg_segment_summary.csv"
records_csv_path = "/Users/housiyuan/project/medical/ecg_segments/data_all.csv"
admissions_csv_path = "/Users/housiyuan/project/medical/ecg_segments/ADMISSIONS.csv"
icu_csv_path = "/Users/housiyuan/project/medical/ecg_segments/ICUSTAYS.csv"
patient_csv_path = "/Users/housiyuan/project/medical/ecg_segments/PATIENTS.csv"
output_csv_path = "/Users/housiyuan/project/medical/ecg_segments/filtered_records.csv"
data_root_dir = "/Volumes/EXTERNAL_USB/mimic-新版cx"

# === 读取数据 ===
base_df = pd.read_csv(base_csv_path)
records_df = pd.read_csv(records_csv_path)
admit_df = pd.read_csv(admissions_csv_path)
icustay_df = pd.read_csv(icu_csv_path)
patient_df = pd.read_csv(patient_csv_path)

# 统一字段名
records_df['SUBJECT_ID'] = records_df['SUBJECT_ID'].astype(int)
records_df['ICUSTAY_ID'] = records_df['ICUSTAY_ID'].astype(int)
base_df['subject_id'] = base_df['subject_id'].astype(int)

tqdm.pandas(desc="Correcting ECG times")
base_df['ecg_time'] = base_df.progress_apply(lambda ecg_row: corrected_ecg_time(ecg_row, data_root_dir), axis=1)

# 转换时间列为datetime
base_df['ecg_time'] = pd.to_datetime(base_df['ecg_time'])
admit_df['ADMITTIME'] = pd.to_datetime(admit_df['ADMITTIME'])
admit_df['DISCHTIME'] = pd.to_datetime(admit_df['DISCHTIME'])
admit_df['DEATHTIME'] = pd.to_datetime(admit_df['DEATHTIME'], errors='coerce')

# 为拼接结果准备新列
merged_rows = []
error_cnt = 0
fuzzy_cnt = 0
fallback_cnt = 0

for idx, row in tqdm(base_df.iterrows(), total=len(base_df)):
    sid = row['subject_id']
    ecg_time = pd.to_datetime(row['ecg_time'])  # 确保时间类型

    # 该患者所有住院记录
    admissions = admit_df[admit_df['SUBJECT_ID'] == sid]

    # 1. 正常匹配：ecg_time 落在 ADMITTIME ~ DISCHTIME 之间
    matched = admissions[
        (admissions['ADMITTIME'] <= ecg_time) &
        (ecg_time < admissions['DISCHTIME'])
    ]

    # 模糊匹配 + 回退匹配
    if matched.empty:
        fuzzy_matched = admissions[
            ((admissions['ADMITTIME'] - ecg_time).abs() <= timedelta(days=7)) |
            ((admissions['DISCHTIME'] - ecg_time).abs() <= timedelta(days=7))
        ]
        if not fuzzy_matched.empty:
            fuzzy_cnt += 1
            matched = fuzzy_matched.sort_values(by='ADMITTIME').head(1)
            delta_admit = (matched['ADMITTIME'] - ecg_time).abs().iloc[0]
            delta_discharge = (matched['DISCHTIME'] - ecg_time).abs().iloc[0]
            print(f"[模糊匹配] subject id: {sid}, ecg_time: {ecg_time}, 和住院时间的差距为 {delta_admit} 和 {delta_discharge}")
        else:
            rec_match = records_df[records_df['SUBJECT_ID'] == sid]
            if rec_match.empty:
                print(f"[ERROR] subject {sid} 在 records 中未找到")
                error_cnt += 1
                continue
            icu_ids = rec_match['ICUSTAY_ID'].dropna().unique()
            icu_match = icustay_df[icustay_df['ICUSTAY_ID'].isin(icu_ids)]
            if icu_match.empty or pd.isna(icu_match.iloc[0]['HADM_ID']):
                print(f"[ERROR] subject {sid} ICU→HADM 匹配失败")
                error_cnt += 1
                continue
            hadm_id = icu_match.iloc[0]['HADM_ID']
            adm_row = admit_df[admit_df['HADM_ID'] == hadm_id]
            if adm_row.empty:
                print(f"[ERROR] subject {sid} HADM_ID={hadm_id} 在admissions中无匹配")
                error_cnt += 1
                continue

            adm_row = adm_row.iloc[0]
            delta_admit = abs((adm_row['ADMITTIME'] - ecg_time).days)
            delta_discharge = abs((adm_row['DISCHTIME'] - ecg_time).days)

            if delta_admit > 28 and delta_discharge > 28:
                print(
                    f"[ERROR] subject {sid}: fallback matched admission too far from ECG time (>{max(delta_admit, delta_discharge)} days)")
                error_cnt += 1
                continue

            matched = pd.DataFrame([adm_row])
            fallback_cnt += 1
            print(f"[回退匹配] subject {sid} 通过 ICU→HADM 找到住院信息，和住院时间的差距为 {delta_admit} 和 {delta_discharge}")

    elif len(matched) > 1:
        print(f"[多条匹配] subject {sid} 有多个住院记录命中，取最早一条")
        matched = matched.sort_values(by='ADMITTIME').head(1)

    admit_info = matched.iloc[0][['HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG']]
    merged_row = pd.concat([row, admit_info])
    merged_rows.append(merged_row)

merged = pd.DataFrame(merged_rows)
final_df = merged.merge(
    patient_df[['SUBJECT_ID', 'DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN', 'EXPIRE_FLAG']],
    left_on='subject_id',
    right_on='SUBJECT_ID',
    how='left'
)
final_df.drop(columns='SUBJECT_ID', inplace=True)

# 排序和标号
final_df = final_df.sort_values(by='subject_id').reset_index(drop=True)
final_df.insert(0, 'ecg_idx', range(1, len(final_df) + 1))
final_df.to_csv(output_csv_path, index=False)

print(f"✅ 完成合并：共 {len(final_df)} 条记录")
print(f"⚠️ 模糊匹配记录数：{fuzzy_cnt}")
print(f"🔄 回退 ICU→HADM 匹配数：{fallback_cnt}")
print(f"❌ 无法匹配记录数（已跳过）：{error_cnt}")