import wfdb
import numpy as np
import matplotlib.pyplot as plt

# 读取记录（自动寻找 .hea 和 .dat）
# record = wfdb.rdrecord('../resources/3652693_0005', pn_dir=None)
record = wfdb.rdrecord('/Users/housiyuan/project/medical/ecg_segments/files_std/10674', pn_dir=None)

print("Sample rate:", record.fs)
print("Lead num:", record.n_sig)
print("sig_name:", record.sig_name)
print("units:", record.units)
print("signal shape:", record.p_signal.shape)  # (time, channels)

data = record.p_signal
start_time = 0
end_time = start_time + 10

if np.isnan(data[start_time*record.fs:end_time*record.fs, :]).any():
    nan_indices = np.argwhere(np.isnan(data[start_time*record.fs:end_time*record.fs, :]))
    for idx in nan_indices:
        print(f"NaN at index: {tuple(idx)}")

ban_list = []
for i in range(len(record.sig_name)):
    if i not in ban_list:
        plt.plot(data[start_time*record.fs:end_time*record.fs, i], label=record.sig_name[i])

plt.legend()
plt.title("ECG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()
