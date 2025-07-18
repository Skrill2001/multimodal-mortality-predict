"""Signal processing code specific to MIMIC-IV-ECG v1.0."""

import os

import pandas as pd

from preprocess import (
    get_pipeline_parser,
    pipeline,
    postprocess_wfdb,
)
from fairseq_signals.utils.file import filenames_from_paths

SOURCE = "mimic_iv_ecg_sur"

def postprocess_mimic_iv_ecg_meta(meta: pd.DataFrame):
    # MIMIC-IV-ECG has no useful comments besides subject ID, which is already recorded
    # Similarly, the date and time are already recorded
    meta = meta.drop(["comments", "base_date", "base_time"], axis=1)

    # General post processing
    return postprocess_wfdb(meta)

def main(args):
    records = pd.read_csv("/ssd/housy/project/fairseq-signals/data/filter_data_sur/filtered_records.csv")
    records["path"] = "/ssd/housy/dataset/" + records["file_name"].str.replace(
        'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0',
        'mimic-iv-ecg')

    records["save_file"] = (SOURCE + "_") + filenames_from_paths(records["path"], replacement_ext=".mat")
    records["source"] = SOURCE
    records["dataset"] = SOURCE
    records.rename(columns={'Unnamed: 0': 'ecg_idx'}, inplace=True)

    pipeline(
        args,
        records,
        SOURCE,
        postprocess_extraction={'meta': postprocess_mimic_iv_ecg_meta},
    )

if __name__ == "__main__":
    parser = get_pipeline_parser()
    args = parser.parse_args()
    main(args)
