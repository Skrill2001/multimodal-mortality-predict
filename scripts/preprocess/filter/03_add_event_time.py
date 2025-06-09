# 主要用来在records中更新关于生存模型建模的时间信息

import pandas as pd
import numpy as np
import os
from datetime import datetime, time

save_dir = './data/mimic_iv_ecg_sur'

# 读取所有CSV文件
records = pd.read_csv('/ssd/housy/project/fairseq-signals/data/mimic_iv_ecg_sur/meta.csv', parse_dates=['dod', 'ecg_time', 'dischtime', 'deathtime'])

os.makedirs(save_dir, exist_ok=True)
print(f'records length: {len(records)}')
print("Load csv files successfully!")
print('-------------------------------------------------------------------------')


# 这里总共可能筛两种数据，一种是在家死的，一种是28天后死的，分别用标志位标注出来

def calculate_survival_metrics(row):
   # 院内死亡情况
    if row['hospital_expire_flag'] == 1 and not pd.isnull(row['deathtime']):
        event_time = row['deathtime']
        event = 1
    # 出院死亡情况，或者院内死亡的遗漏数据
    elif not pd.isna(row['dod']):
        event_time = datetime.combine(row['dod'], time(12, 00))
        event = 1
    else:
        event_time = row['dischtime']
        event = 0
    
    delta_hours = (event_time - row['ecg_time']).total_seconds() / 3600
    return pd.Series([event, event_time, delta_hours])

records[['event', 'event_time', 'delta_hours']] = records.apply(calculate_survival_metrics, axis=1)
records = records[records['delta_hours'] > 0.0]

# 4. 离散时间窗划分
time_bins = [0, 24, 72, 168, 336, 504]  # 单位：小时
bin_labels = ['0-24h', '24-72h', '72h-7d', '7d-14d', '14d-21d']
last_day = int(time_bins[-1]/24)

records['time_bin'] = pd.cut(
    records['delta_hours'],
    bins=time_bins,
    labels=bin_labels,
    right=False
)

# 构建标签列表（每个元素是长度为7的数组）
records['label_list'] = records.apply(lambda x: [0]*len(bin_labels), axis=1)
for i, (lower, upper) in enumerate(zip(time_bins[:-1], time_bins[1:])):
    
    mask = (records['event'] != 0 ) & (records['delta_hours'] >= lower) & (records['delta_hours'] < upper)
    records.loc[mask, 'label_list'] = records.loc[mask]['label_list'].apply(
        lambda lst: [1 if j == i else lst[j] for j in range(len(lst))]
    )
records.loc[records['delta_hours'] >= time_bins[-1], 'event'] = 0

# 添加用于筛选的标志位
records[f'died_after_{last_day}d'] = np.where((~records['dod'].isna()) & (records['delta_hours'] >= time_bins[-1]), True, False)
records['died_in_home'] = np.where((~records['dod'].isna()) & (records['hospital_expire_flag'] == 0), True, False)
records[f'died_in_home_after_{last_day}d'] = np.where(records[f'died_after_{last_day}d'] & records['died_in_home'], True, False)

count_died_after_last_day = (records[f'died_after_{last_day}d'] == True).sum()
count_died_in_home = (records['died_in_home'] == True).sum()
count_died_in_home_after_last_day = (records[f'died_in_home_after_{last_day}d'] == True).sum()
# count_filter = ((records['died_after_28d'] == True) & (records['died_in_home'] == True)).sum()
print(f"died afer {last_day} days: {count_died_after_last_day}, died in home: {count_died_in_home}, died in home after {last_day}d: {count_died_in_home_after_last_day}")

# 计算各时间段死亡率
mortality_stats = []
for i, bin_label in enumerate(bin_labels):
    lower = time_bins[i]
    upper = time_bins[i+1]
    
    in_window = (records['time_bin'] == bin_label) 
    deaths_in_window = records[in_window & (records['event'] == 1)].shape[0]
    deaths_in_home = records[in_window & (records['event'] == 1) & (records['hospital_expire_flag'] == 0)].shape[0]
    deaths_in_hosp = records[in_window & (records['event'] == 1) & (records['hospital_expire_flag'] == 1)].shape[0]
    total_in_window = records[in_window | ((records['event'] == 1) & (records['delta_hours'] >= upper)) | (records['event'] == 0)].shape[0]
    
    mortality_rate = deaths_in_window / total_in_window if total_in_window > 0 else 0
    mortality_rate_hosp = deaths_in_hosp / total_in_window if total_in_window > 0 else 0
    mortality_rate_home = deaths_in_home / total_in_window if total_in_window > 0 else 0
    
    mortality_stats.append({
        'time_window': bin_label,
        'deaths': deaths_in_window,
        'deaths in hosp': deaths_in_hosp,
        'deaths in home': deaths_in_home,
        'total_ecgs': total_in_window,
        'mortality_rate': f"{mortality_rate:.2%}",
        'mortality_rate_hosp': f"{mortality_rate_hosp:.2%}",
        'mortality_rate_home': f"{mortality_rate_home:.2%}"
    })


# 打印统计结果
print("各时间段死亡率统计：")
print(pd.DataFrame(mortality_stats))

total_num = len(records[records[f'died_in_home_after_{last_day}d']==False])
death_num = len(records[ (records['hospital_expire_flag']==1) & (records['event']==1) & (records[f'died_after_{last_day}d']==False)])
print(f"\n处理完成！有效样本数：{total_num}, {last_day}天内住院死亡总数：{death_num}, {last_day}天内样本总死亡率：{death_num/total_num:.4f}")


# 删除不需要的列
cols_to_drop = [f'mean_{i}' for i in range(12)] + [f'std_{i}' for i in range(12)]
records.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# ----------------------- 保存过滤后的文件 -----------------------
records.to_csv(os.path.join(save_dir, 'meta_add_event_5.csv'), index=False)
pd.DataFrame(mortality_stats).to_csv(os.path.join(save_dir, 'mortality_statistics_5.csv'), index=False)
print(f"Data is saved in {save_dir}")
