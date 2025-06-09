import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from fairseq_signals.models import build_model_from_checkpoint
from fairseq_signals.data.ecg.raw_ecg_dataset import ECGSurvivalDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay, RocCurveDisplay
import json
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from pycox.evaluation import concordance
from pycox.evaluation import EvalSurv
from model import find_optimal_threshold, find_checkpoint, FineTunedWav2Vec2Model_concat, FineTunedWav2Vec2Model_full

run_name = 'sur-fintune-lr-1e-6-weighted-2layer-long'
epoch = '197'

middle_dim = 768
dropout = 0.3

threshold = 0.3
threshold_method = 'youden'
batch_size = 256

# time_points = [6, 12, 24, 48, 72, 168, 336, 672]
# bin_names = ['0-6h', '6-12h', '12-24h', '24-48h', '48-72h', '72h-7d', '7d-14d', '14d-28d']

time_points = [720, 2160, 4320, 8640, 25920, 43200, 86400]
time_bins = [0, 720, 2160, 4320, 8640, 25920, 43200, 86400]
bin_names = ['0-1m', '1-3m', '3-6m', '6-12m', '1-3y', '3y-5y', '5y-10y']

# time_points = [24, 72, 168, 336, 504]
# bin_names = ['0-24h', '24-72h', '72h-7d', '7d-14d', '14d-21d']

# pretrained_path只是为了构建model用，实际权重全都在ckpt_path里面
pretrained_path = "ckpts/pretrained/ecg_fm/physionet_finetuned.pt"
ckpt_dir = f'ckpts/{run_name}'
ckpt_path, ckpt_name = find_checkpoint(ckpt_dir, epoch)
records_path = "./data/mimic_iv_ecg_sur/meta_split_long.csv"
data_dir = './data/mimic_iv_ecg_sur/segmented'

save_info_dir = f'outputs/{run_name}/{ckpt_name}'
print(f'ckpt name is [{ckpt_name}]')
os.makedirs(save_info_dir, exist_ok=True)

# GPU and model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_dataset = ECGSurvivalDataset(
    split="test",
    records_path=records_path,
    data_dir=data_dir,
    sample_rate=500,
    target_size=2500,
    time_bins=time_bins
)

# Create TensorDataset
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=8, shuffle=False)

class FineTunedWav2Vec2Model(nn.Module):
    def __init__(self, pretrained_model, middle_dim, dropout, num_label_bins):

        super(FineTunedWav2Vec2Model, self).__init__()

        self.pretrained_model = pretrained_model
        self.encoder_embed_dim = self.pretrained_model.proj.in_features
    
        self.proj = nn.Sequential(
            # nn.Dropout(dropout), 
            nn.Linear(2 * self.encoder_embed_dim, self.encoder_embed_dim),
            nn.SiLU(),
            # nn.Linear(self.encoder_embed_dim, middle_dim),
            # nn.SiLU(),
            nn.Linear(middle_dim, num_label_bins)
        )

    def get_features(self, x):
        res = self.pretrained_model(source=x)
        return res["encoder_out"].mean(dim=1)

    def forward(self, segment_0, segment_1):
        
        feature_0 = self.get_features(segment_0)
        feature_1 = self.get_features(segment_1)

        combined_feature = torch.cat([feature_0, feature_1], dim=-1)
        logits = self.proj(combined_feature)
        return logits


# Model initialization
model_pretrained = build_model_from_checkpoint(checkpoint_path=(os.path.join(pretrained_path))).to(device)
model_with_classification_head = FineTunedWav2Vec2Model(pretrained_model=model_pretrained, middle_dim=middle_dim, dropout=dropout, num_label_bins=len(bin_names))

# Loads the fine-tuned model weights from train.py.
model_with_classification_head.load_state_dict(torch.load(ckpt_path, weights_only=True))
model_with_classification_head.to(device)

# Evaluate with validation data
model_with_classification_head.eval()

# Performance evaluation and confusion matrix visualization
def evaluate_model(model, dataloader, dataset_name, device):

    model.eval()
    all_probs = []
    all_surv_probs = []
    all_event_idx = []
    all_delta_hours = []
    all_events = []

    class_names=['Alive', 'Death']

    metrics = {"Dataset": dataset_name, 'to_time_stat': {}, 'time_interval_stat': {}}

    with torch.no_grad():

        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            segment_0 = batch["segment_0"].to(device)
            segment_1 = batch["segment_1"].to(device)
            event_idx = batch['duration_idx']
            events = batch['event']
            delta_hours = batch['delta_hours']
            
            outputs = model(segment_0=segment_0, segment_1=segment_1)
            logits = outputs.squeeze()
            probs = torch.sigmoid(logits)
            surv_prob = torch.cumprod(1 - probs, dim=1)
            
            all_probs.append(probs.cpu())
            all_surv_probs.append(surv_prob.cpu())
            all_event_idx.append(event_idx)
            all_delta_hours.append(delta_hours)
            all_events.append(events)

        y_probs = torch.cat(all_probs).numpy()  # (N,8)
        y_surv_probs = torch.cat(all_surv_probs).numpy()
        y_event_idx = torch.cat(all_event_idx).numpy()
        y_delta_hours = torch.cat(all_delta_hours).numpy()
        y_events = torch.cat(all_events).numpy()

        print("累积区间计算: ")
        for timepoint in time_points:
            print(f"≤{timepoint}h")
            timepoint_result = {}
            if timepoint != time_points[-1]:
                mask = (y_delta_hours <= timepoint)  # 包括删失样本（需处理）
                y_true = (y_events[mask] == 1).astype(int)
                y_score = 1 - y_surv_probs[mask, y_event_idx[mask]]
            else:
                y_true = y_events.astype(int)
                y_score = 1 - y_surv_probs[:, -1]

            if len(np.unique(y_true)) >= 2:  # 确保有正负样本
                
                auc = roc_auc_score(y_true, y_score)
                timepoint_result['auc'] = auc
                RocCurveDisplay.from_predictions(y_true, y_score)
                plt.savefig(os.path.join(save_info_dir, f'roc_curve_{timepoint}h.png'), bbox_inches='tight', dpi=300)

                threshold_youden, metrics_youden = find_optimal_threshold(y_true, y_score, method=threshold_method)
                print(f"最优阈值（{threshold_method}）: {threshold_youden:.4f}, specificity: {metrics_youden['specificity']:.4f}")

                y_pred = (y_score >= threshold_youden)
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                acc = (tp + tn) / (tp + tn + fp + fn)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                timepoint_result['Accuracy'] = acc
                timepoint_result['Recall'] = recall
                print(f'AUC: {auc:.4f}, Accuracy: {acc:.4f}, Recall: {recall:.4f}')

                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
                disp.plot(cmap="Blues", values_format="d")
                plt.title("Binary Confusion Matrix")
                plt.xlabel("Predicted label")
                plt.ylabel("True label")
                plt.savefig(os.path.join(save_info_dir, f"binary_confusion_matrix_to_{timepoint}h.png"), bbox_inches="tight", dpi=300)
                plt.close()

            metrics['to_time_stat'][timepoint] = timepoint_result
        print('----------------------------------------------------------------')

        print("分区间计算: ")
        for i, bin_name in enumerate(bin_names):
            print(bin_name)
            bin_result = {}
            event_mask = (y_event_idx == i) & (y_events == 1)
            # 当前区间未发生事件的样本（包括删失和其他区间事件）
            non_event_mask = (y_event_idx > i) | (y_events == 0)
            
            valid_mask = event_mask | non_event_mask
            y_true_bin = event_mask[valid_mask].astype(int)
            # y_prob_bin = y_probs[valid_mask, i]
            y_prob_bin = 1 - y_surv_probs[valid_mask, i]
            
            if len(np.unique(y_true_bin)) > 1:
                auc = roc_auc_score(y_true_bin, y_prob_bin)
                RocCurveDisplay.from_predictions(y_true_bin, y_prob_bin)
                plt.savefig(os.path.join(save_info_dir, f'roc_curve_{bin_name}.png'), bbox_inches='tight', dpi=300)
                bin_result['AUC'] = auc
                
                threshold_youden, metrics_youden = find_optimal_threshold(y_true_bin, y_prob_bin, method=threshold_method)
                print(f"最优阈值（{threshold_method}）: {threshold_youden:.4f}, specificity: {metrics_youden['specificity']:.4f}")

                y_pred_bin = (y_prob_bin >= threshold_youden)
                cm = confusion_matrix(y_true_bin, y_pred_bin)
                tn, fp, fn, tp = cm.ravel()
                acc = (tp + tn) / (tp + tn + fp + fn)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                bin_result['Accuracy'] = acc
                bin_result['Recall'] = recall
                print(f'AUC: {auc:.4f}, Accuracy: {acc:.4f}, Recall: {recall:.4f}')

                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
                disp.plot(cmap="Blues", values_format="d")
                plt.title("Binary Confusion Matrix")
                plt.xlabel("Predicted label")
                plt.ylabel("True label")
                plt.savefig(os.path.join(save_info_dir, f"binary_confusion_matrix_to_{bin_name}h.png"), bbox_inches="tight", dpi=300)
                plt.close()
            
            metrics['time_interval_stat'][bin_name] = bin_result

        print('----------------------------------------------------------------')

        surv_df = pd.DataFrame(y_surv_probs, columns=time_points)
        evaluator = EvalSurv(surv_df.T, y_delta_hours, y_events, censor_surv='km')
        c_index_value = evaluator.concordance_td('antolini')
        print(f"C-index: {c_index_value:.4f}")
        metrics['C_index'] = c_index_value
        print('----------------------------------------------------------------')

        # 1. 生成真实标签（多分类）; 死亡样本：使用 event_idx 作为类别 ; 存活样本：类别 8
        y_true_multiclass = np.where(y_events == 1, y_event_idx, len(bin_names))

        # 2. 生成预测标签, 预测概率最高的区间下标, 所有区间都小于0.5那就算活着
        y_pred_multiclass = np.argmax(y_probs, axis=1)
        y_pred_multiclass[(y_probs.max(axis=1) < threshold)] = len(time_points)

        # 3. 计算混淆矩阵
        bin_names_with_survival = bin_names + ["Survival"]
        cm = confusion_matrix(y_true_multiclass, y_pred_multiclass, labels=range(len(bin_names)+1))

        # 4. 可视化
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=bin_names_with_survival
        )
        disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
        plt.title(f"Confusion Matrix ({dataset_name})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_info_dir, f"confusion_matrix_multiclass_{dataset_name}.png"), bbox_inches='tight', dpi=300)
        plt.close()

        # 5. 打印分类报告
        print("\nClassification Report (9 class):")
        print(classification_report(
            y_true_multiclass,
            y_pred_multiclass,
            target_names=bin_names_with_survival,
            digits=4
        ))

        print('----------------------------------------------------------------')

        # 生成预测标签
        y_pred = np.zeros(len(y_events))
        y_prob = np.zeros(len(y_events))
        
        # 规则1: 事件样本 -> 检查事件所在区间的死亡风险
        event_mask = (y_events == 1)
        y_prob[event_mask] = (1 - y_surv_probs[event_mask, y_event_idx[event_mask]])
        
        # 规则2: 删失样本 -> 检查最后区间的死亡风险
        censored_mask = (y_events == 0)
        y_prob[censored_mask] = (1 - y_surv_probs[censored_mask, y_event_idx[censored_mask]])
        
        
        # 计算混淆矩阵
        auc = roc_auc_score(y_events, y_prob)
        print('auc:', auc)

        threshold_youden, metrics_youden = find_optimal_threshold(y_true, y_score, method=threshold_method)
        print(f"最优阈值（{threshold_method}）: {threshold_youden:.4f}, specificity: {metrics_youden['specificity']:.4f}")

        y_pred = y_prob >= threshold_youden
        cm = confusion_matrix(y_events, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 可视化
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Binary Confusion Matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.savefig(os.path.join(save_info_dir, "binary_confusion_matrix.png"), bbox_inches="tight", dpi=300)
        plt.close()
        
        cm_metrics = {
            'Accuracy': (tp + tn) / (tp + tn + fp + fn),
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,        # 召回率/真正率: 预测正确的所有正样本占实际所有正样本的比例
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,   # 真负率: 预测正确的所有负样本占实际所有负样本的比例
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,     # 精确率: 预测所有正样本中判断正确的比例
            'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,           # 阴性精确率: 预测所有负样本中判断正确的比例
            'F1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,  # F1-score
        }
        metrics['cm_metrics'] = cm_metrics

        print("\nConfusion Matrix (2 class):\n", pd.DataFrame(cm, index=[f"True {name}" for name in class_names],
                                                                columns=[f"Pred {name}" for name in class_names]))

        # Save performance metrics in a dictionary
        return metrics


# Save evaluation results for each test set
def save_metrics_to_file(model, device):
    results = []
    results.append(evaluate_model(model, test_loader, "Internal_Test_Set", device))

    save_metrics_path = os.path.join(save_info_dir, "test_set_metrics.json")
    with open(save_metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nPerformance metrics have been saved to {save_metrics_path}.")

save_metrics_to_file(model_with_classification_head, device)
