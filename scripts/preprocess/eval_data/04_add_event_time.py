# update time info in survival modeling

import pandas as pd
import numpy as np
import os
from datetime import datetime, time

save_dir = "/Users/housiyuan/project/medical/ecg_segments/"
record_csv_path = "/Users/housiyuan/project/medical/ecg_segments/filtered_records.csv"
records = pd.read_csv(record_csv_path, parse_dates=['ecg_time', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DOD', 'DOD_SSN'])

print(f'records length: {len(records)}')
print("Load csv files successfully!")
print('-------------------------------------------------------------------------')


# mark died in home and died after 28 days
def calculate_survival_metrics(row):
   # died in hosp
    if row['HOSPITAL_EXPIRE_FLAG'] == 1 and not pd.isnull(row['DEATHTIME']):
        event_time = row['DEATHTIME']
        event = 1
    # died in home
    elif not pd.isna(row['DOD']):
        event_time = datetime.combine(row['DOD'], time(12, 00))
        event = 1
    else:
        event_time = row['DISCHTIME']
        event = 0
    
    delta_hours = (event_time - row['ecg_time']).total_seconds() / 3600
    return pd.Series([event, event_time, delta_hours])

records[['event', 'event_time', 'delta_hours']] = records.apply(calculate_survival_metrics, axis=1)
records = records[records['delta_hours'] > 0.0]

time_bins = [0, 6, 12, 24, 48, 72, 168, 336, 672]  # 单位：小时
bin_labels = ['0-6h', '6-12h', '12-24h', '24-48h', '48-72h', '72h-7d', '7d-14d', '14d-28d']
last_day = int(time_bins[-1]/24)

records['time_bin'] = pd.cut(
    records['delta_hours'],
    bins=time_bins,
    labels=bin_labels,
    right=False
)

records['label_list'] = records.apply(lambda x: [0]*len(bin_labels), axis=1)
for i, (lower, upper) in enumerate(zip(time_bins[:-1], time_bins[1:])):
    
    mask = (records['event'] != 0 ) & (records['delta_hours'] >= lower) & (records['delta_hours'] < upper)
    records.loc[mask, 'label_list'] = records.loc[mask]['label_list'].apply(
        lambda lst: [1 if j == i else lst[j] for j in range(len(lst))]
    )
records.loc[records['delta_hours'] >= time_bins[-1], 'event'] = 0

records[f'died_after_{last_day}d'] = np.where((~records['DOD'].isna()) & (records['delta_hours'] >= time_bins[-1]), True, False)
records['died_in_home'] = np.where((~records['DOD'].isna()) & (records['HOSPITAL_EXPIRE_FLAG'] == 0), True, False)
records[f'died_in_home_after_{last_day}d'] = np.where(records[f'died_after_{last_day}d'] & records['died_in_home'], True, False)

count_died_after_last_day = (records[f'died_after_{last_day}d'] == True).sum()
count_died_in_home = (records['died_in_home'] == True).sum()
count_died_in_home_after_last_day = (records[f'died_in_home_after_{last_day}d'] == True).sum()
# count_filter = ((records['died_after_28d'] == True) & (records['died_in_home'] == True)).sum()
print(f"died afer {last_day} days: {count_died_after_last_day}, died in home: {count_died_in_home}, died in home after {last_day}d: {count_died_in_home_after_last_day}")

mortality_stats = []
for i, bin_label in enumerate(bin_labels):
    lower = time_bins[i]
    upper = time_bins[i+1]
    
    in_window = (records['time_bin'] == bin_label) 
    deaths_in_window = records[in_window & (records['event'] == 1)].shape[0]
    deaths_in_home = records[in_window & (records['event'] == 1) & (records['HOSPITAL_EXPIRE_FLAG'] == 0)].shape[0]
    deaths_in_hosp = records[in_window & (records['event'] == 1) & (records['HOSPITAL_EXPIRE_FLAG'] == 1)].shape[0]
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


print("Mortality Info：")
print(pd.DataFrame(mortality_stats))

total_num = len(records[records[f'died_in_home_after_{last_day}d']==False])
death_num = len(records[ (records['HOSPITAL_EXPIRE_FLAG']==1) & (records['event']==1) & (records[f'died_after_{last_day}d']==False)])
print(f"\nComplete！valid samples：{total_num}, died in hosp within {last_day} days：{death_num}, mortality within {last_day} days：{death_num/total_num:.4f}")

cols_to_drop = [f'mean_{i}' for i in range(12)] + [f'std_{i}' for i in range(12)]
records.drop(columns=cols_to_drop, inplace=True, errors='ignore')

records.to_csv(os.path.join(save_dir, 'final_data_add_event_8.csv'), index=False)
pd.DataFrame(mortality_stats).to_csv(os.path.join(save_dir, 'mortality_statistics_8.csv'), index=False)
print(f"Data is saved in {save_dir}")
