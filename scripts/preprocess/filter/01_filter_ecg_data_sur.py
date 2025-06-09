import pandas as pd
import os
from tqdm import tqdm

ALL_HADM = True
save_dir = 'data/filter_data_sur'

# 读取所有CSV文件
admissions = pd.read_csv('/mnt/nvme_share/common/datasets/mimic-iv/mimic-iv-2.2/hosp/admissions.csv', parse_dates=['admittime', 'dischtime', 'deathtime'])
records = pd.read_csv('/ssd/housy/dataset/mimic-iv-ecg-ext-icd/records_with_hadm.csv', parse_dates=['ecg_time', 'deathtime', 'dischtime', 'admittime'])
icustays = pd.read_csv('/mnt/nvme_share/common/datasets/mimic-iv/mimic-iv-2.2/icu/icustays.csv', parse_dates=['intime', 'outtime'])
discharge = pd.read_csv('/ssd/housy/dataset/mimic-iv-note/discharge.csv.gz')

os.makedirs(save_dir, exist_ok=True)
print(f'records length: {len(records)}')
print("Load csv files successfully!")
print('-------------------------------------------------------------------------')

# 1. 年龄筛选：只保留18岁及以上患者
valid_age_subjects = records[records['age'] >= 18]['subject_id'].unique()
records = records[records['subject_id'].isin(valid_age_subjects)]
admissions = admissions[admissions['subject_id'].isin(valid_age_subjects)]
icustays = icustays[icustays['subject_id'].isin(valid_age_subjects)]
discharge = discharge[discharge['subject_id'].isin(valid_age_subjects)]

# 2. 住院记录筛选，这个过程不筛人，只筛住院记录
if ALL_HADM:
    # 保留所有住院记录，不做去重
    valid_adm_hadm = admissions['hadm_id'].unique()
else:
    # 保留首次住院记录（按admittime排序取第一条）
    admissions = admissions.sort_values('admittime').groupby('subject_id').first().reset_index()
    valid_adm_hadm = admissions['hadm_id'].unique()

# 3. ICU停留时间筛选（los单位为天），这个过程也不筛人，只筛icu stays
icustays = icustays[icustays['hadm_id'].isin(valid_adm_hadm)]
valid_icu_hadm = icustays['hadm_id'].unique()
icustays = icustays.sort_values(['subject_id', 'hadm_id', 'intime'])

# 4. 心电数据筛选,这个过程可能筛人，因为有的人可能没有ecg
records = records[ (records['ecg_taken_in_ed_or_hosp'] == True) & (records['hadm_id'].isin(valid_adm_hadm)) ]
valid_ecg_hadm = records['hadm_id'].unique()

# 5. 筛选出含有出院/放射报告的subject，这个过程会筛人，也就是人只有有报告和没有报告两种，没有报告的会被筛掉
discharge = discharge[discharge['hadm_id'].isin(valid_adm_hadm)]
valid_discharge_hadm = discharge['hadm_id'].unique()

# 6. 综合筛选
final_hadm = set(valid_ecg_hadm) & set(valid_discharge_hadm) & set(valid_icu_hadm) & set(valid_adm_hadm)
admissions = admissions[admissions['hadm_id'].isin(final_hadm)]
icustays = icustays[icustays['hadm_id'].isin(final_hadm)]
discharge = discharge[discharge['hadm_id'].isin(final_hadm)]
records = records[ (records['hadm_id'].isin(final_hadm))]
print("filter successfully!")

# 7. 患者ID一致性校验
admissions_subjects = set(admissions['subject_id'])
icustays_subjects = set(icustays['subject_id'])
records_subjects = set(records['subject_id'])
discharge_subjects = set(discharge['subject_id'])

if admissions_subjects == icustays_subjects == records_subjects == discharge_subjects:
    print("subjects id evaluation ok!")
else:
    print("subjects id evaluation false!")
    common_subjects = set(admissions['subject_id']) & set(icustays['subject_id']) & set(discharge['subject_id']) & set(records['subject_id'])
    admissions = admissions[admissions['subject_id'].isin(common_subjects)]
    icustays = icustays[icustays['subject_id'].isin(common_subjects)]
    records = records[records['subject_id'].isin(common_subjects)]
    discharge = discharge[discharge['subject_id'].isin(common_subjects)]


# 8 住院ID一致性校验
def get_valid_hadm(df, col_name='hadm_id'):
    return set(df[col_name].dropna().astype(int).unique())

valid_hadm_adm = get_valid_hadm(admissions)
valid_hadm_icu = get_valid_hadm(icustays)
valid_hadm_rec = get_valid_hadm(records)
valid_hadm_dis = get_valid_hadm(discharge)

if valid_hadm_adm == valid_hadm_dis == valid_hadm_icu == valid_hadm_rec:
    print("hadm id evaluation ok!")
else:
    print("hadm id evaluation false!")
    common_hadm = valid_hadm_adm & valid_hadm_icu & valid_hadm_rec & valid_hadm_dis
    admissions = admissions[admissions['hadm_id'].isin(common_hadm)]
    icustays = icustays[icustays['hadm_id'].isin(common_hadm)]
    records = records[records['hadm_id'].isin(common_hadm)]
    discharge = discharge[discharge['hadm_id'].isin(common_hadm)]

# 9. 合并数据 
# 把icu的数据合并上来，如果这个ecg恰好在一个icu内，记录该icu的时间
records['in_icu_stay_id'] = pd.NA
records['intime'] = pd.NaT
records['outtime'] = pd.NaT

# 为了加速后面查找，按 hadm_id 分组 ICU stays
icustays_grouped = icustays.groupby('hadm_id')
for idx, row in tqdm(records.iterrows(), total=len(records)):
    hadm_id = row['hadm_id']
    ecg_time = row['ecg_time']
    
    # 获取这个 hadm_id 的所有 ICU stays
    if hadm_id in icustays_grouped.groups:
        stays = icustays_grouped.get_group(hadm_id)
        
        # 查找 ecg_time 落在哪个 ICU stay 中
        match = stays[(stays['intime'] <= ecg_time) & (stays['outtime'] >= ecg_time)]
        
        if not match.empty:
            first_match = match.iloc[0]
            records.at[idx, 'in_icu_stay_id'] = first_match['stay_id']
            records.at[idx, 'intime'] = first_match['intime']
            records.at[idx, 'outtime'] = first_match['outtime']


# 新增列：计算 ECG 数据与临床终点（死亡或出院）之间的时间差（小时）
def get_endpoint_time(row):
    if row['hospital_expire_flag'] == 1 and pd.notnull(row['deathtime']):
        return row['deathtime']
    elif pd.notnull(row['dischtime']):
        return row['dischtime']
    else:
        return pd.NaT

records['endpoint_time'] = records.apply(get_endpoint_time, axis=1)
records['time_to_endpoint_hours'] = (records['endpoint_time'] - records['ecg_time']).dt.total_seconds() / 3600


# 新增列：判断 time_to_endpoint_hours 是否在 0.5 到 2 小时之间
records['within_0.5_2h'] = records['time_to_endpoint_hours'].apply(
    lambda x: True if pd.notnull(x) and 0.5 <= x <= 2 else False
)

# 删除不需要的列
cols_to_drop = ['ed_diag_ed', 'ed_diag_hosp', 'hosp_diag_hosp', 
                'all_diag_hosp', 'all_diag_all', 'fold', 'strat_fold']
records.drop(columns=cols_to_drop, inplace=True, errors='ignore')

print("Data process successfully!")
print('-------------------------------------------------------------------------')

num_ecg = len(records)
num_patient = len(records['subject_id'].unique())
num_admission = len(admissions)
num_in_icu = records['in_icu_stay_id'].notna().sum()
num_death_ecg = records['hospital_expire_flag'].sum()
num_within_range = records['within_0.5_2h'].sum()


print(f"ECG 总数量: {num_ecg}")
print(f"病人 总数量: {num_patient}")
print(f"住院 总数量: {num_admission}")
print(f"ECG 在 ICU 内数量: {num_in_icu}")
print(f"总的死亡ECG条数: {num_death_ecg}")
print(f"ECG 距离临床终点在 0.5~2 小时范围内数量: {num_within_range}")


# ----------------------- 保存过滤后的文件 -----------------------
admissions.to_csv(os.path.join(save_dir, 'filtered_admissions.csv'), index=False)
icustays.to_csv(os.path.join(save_dir, 'filtered_icustays.csv'), index=False)
records.to_csv(os.path.join(save_dir, 'filtered_records.csv'), index=False)
discharge.to_csv(os.path.join(save_dir, 'filtered_discharge.csv'), index=False)

print(f"Data save in {save_dir}")