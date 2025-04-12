# 将nan数据筛选后的信息更新到其他几个文件中

import pandas as pd
import os

save_dir = './data/filter_data_final'

# 读取所有CSV文件
admissions = pd.read_csv('/ssd/housy/project/fairseq-signals/data/filter_data/filtered_admissions.csv')
records = pd.read_csv('/ssd/housy/project/fairseq-signals/data/mimic_iv_ecg/meta.csv')
icustays = pd.read_csv('/ssd/housy/project/fairseq-signals/data/filter_data/filtered_icustays.csv')
discharge = pd.read_csv('/ssd/housy/project/fairseq-signals/data/filter_data/filtered_discharge.csv')

os.makedirs(save_dir, exist_ok=True)
print("Load csv files successfully!")
print('-------------------------------------------------------------------------')


# 1. 按照eeg信息进行筛选
original_ecg_num = len(records)
filtered_out_num = len(records[ (records['nan_any'] == True) | (records['constant_leads_any'] == True) ]['ecg_idx'].unique())
print("有nan值的ECG数量: ", len(records[records['nan_any'] == True]['ecg_idx'].unique()))
print("有恒定lead值的ECG数量: ", len(records[records['constant_leads_any'] == True]['ecg_idx'].unique()))
print("总共排除的ECG数量: ", filtered_out_num)

# 一次住院可能有多个ecg，但可能不是都坏，所以不能用hadm筛，对于ecg还是要用ecg_idx，对于其他的数据用hadm_id就可以
valid_ecg_idx = records[ (records['nan_any'] == False) & (records['constant_leads_any'] == False) ]['ecg_idx'].unique()
valid_ecg_hadm = records[ (records['nan_any'] == False) & (records['constant_leads_any'] == False) ]['hadm_id'].unique()
records = records[records['ecg_idx'].isin(valid_ecg_idx)]
admissions = admissions[admissions['hadm_id'].isin(valid_ecg_hadm)]
icustays = icustays[icustays['hadm_id'].isin(valid_ecg_hadm)]
discharge = discharge[discharge['hadm_id'].isin(valid_ecg_hadm)]

filtered_ecg_num = len(records)
if original_ecg_num - filtered_ecg_num == filtered_out_num:
    print("filter successfully!")
else:
    print(f"filter failed. original ecg num is {original_ecg_num}, filtered ecg num is {filtered_ecg_num}, filtered out num is {filtered_out_num}.")
    exit()
print('-------------------------------------------------------------------------')

flag_subject_eval = False
flag_hadm_eval = False

# 7. 患者ID一致性校验
admissions_subjects = set(admissions['subject_id'])
icustays_subjects = set(icustays['subject_id'])
records_subjects = set(records['subject_id'])
discharge_subjects = set(discharge['subject_id'])

if admissions_subjects == icustays_subjects == records_subjects == discharge_subjects:
    flag_subject_eval = True
    print("subjects id evaluation ok!")
else:
    print("subjects id evaluation false!")


# 8 住院ID一致性校验
def get_valid_hadm(df, col_name='hadm_id'):
    return set(df[col_name].dropna().astype(int).unique())

valid_hadm_adm = get_valid_hadm(admissions)
valid_hadm_icu = get_valid_hadm(icustays)
valid_hadm_rec = get_valid_hadm(records)
valid_hadm_dis = get_valid_hadm(discharge)

if valid_hadm_adm == valid_hadm_dis == valid_hadm_icu == valid_hadm_rec:
    flag_hadm_eval = True
    print("hadm id evaluation ok!")
else:
    print("hadm id evaluation false!")

if flag_subject_eval and flag_hadm_eval:
    print("Data evaluation successfully!")
else:
    print("Data evaluation failed!")
    exit()
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

print(f"Data is saved in {save_dir}")