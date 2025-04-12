import pandas as pd

# 假设 records 是已经读入的 DataFrame
record_i = pd.read_csv('/ssd/housy/dataset/mimic-iv-ecg-ext-icd/records_w_diag_icd10.csv')

def resolve_hadm(records):
    # 初始化 hadm_id 列，默认设置为 NaN
    records['hadm_id'] = pd.NA

    # 条件 1: 只有 ed_hadm_id 有值
    only_ed = records['ed_hadm_id'].notna() & records['hosp_hadm_id'].isna()
    records.loc[only_ed, 'hadm_id'] = records.loc[only_ed, 'ed_hadm_id']

    # 条件 2: 只有 hosp_hadm_id 有值
    only_hosp = records['hosp_hadm_id'].notna() & records['ed_hadm_id'].isna()
    records.loc[only_hosp, 'hadm_id'] = records.loc[only_hosp, 'hosp_hadm_id']

    # 条件 3: 两个都有值且相等
    both_equal = (records['ed_hadm_id'].notna() & 
                records['hosp_hadm_id'].notna() & 
                (records['ed_hadm_id'] == records['hosp_hadm_id']))
    records.loc[both_equal, 'hadm_id'] = records.loc[both_equal, 'ed_hadm_id']

    # 条件 4: 两个都有值但不相等 -> 删除这些行
    conflict_rows = (records['ed_hadm_id'].notna() &
                    records['hosp_hadm_id'].notna() &
                    (records['ed_hadm_id'] != records['hosp_hadm_id']))
    num_conflict = conflict_rows.sum()

    # 输出冲突行数
    print(f'存在冲突的记录数（已删除）: {num_conflict}')

    # 删除冲突行
    records = records[~conflict_rows].reset_index(drop=True)

    return records

print(len(record_i))
records_o = resolve_hadm(record_i)
print(len(records_o))

records_o.to_csv('/ssd/housy/dataset/mimic-iv-ecg-ext-icd/records_with_hadm.csv')