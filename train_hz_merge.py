import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from fairseq_signals.models import build_model_from_checkpoint
from fairseq_signals.data.ecg.raw_ecg_dataset import ECGSurvivalDataset, SurvivalPredictDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5EncoderModel, T5Config
import glob
import pandas as pd

from pycox.models.loss import NLLLogistiHazardLoss
from pycox.evaluation import EvalSurv
from model import WeightedNLLLogistiHazardLoss, find_optimal_threshold, FineTunedWav2Vec2Model_full, FineTunedWav2Vec2Model_concat
from sklearn.metrics import confusion_matrix

from tabnet import TabNetWithEmbed
from t5 import ClinicalT5MortalityPredictor

run_name = 'fusion-finetune-lr-1e-6-2layer-long-bs-64'

ecg_base_checkpoint_path = 'ckpts/pretrained/ecg_fm/physionet_finetuned.pt'
ecg_pretrained_checkpoint_path = 'ckpts/sur-fintune-lr-1e-6-weighted-2layer/best-acc-0.75-epoch-198.pth'

tabnet_checkpoint_path = 'ckpts/pretrained/tabnet/512_best_auc_model.pth'
t5_checkpoint_path = 'ckpts/pretrained/Clinical-T5-Base'

records_path = "./data/combined_data/meta_split_died.csv"
data_dir = './data/combined_data/segmented'

save_ckpt_dir = f'ckpts/{run_name}'
save_info_dir = f'outputs/{run_name}'

hidden_dim = 512
dropout = 0.3

num_epochs = 100
batch_size = 64

learning_rate = 1e-6
betas = (0.9, 0.98)
weight_decay = 1e-5

FREEZE_PRETRAINED_MODEL = True

time_points = np.array([720, 2160, 4320, 8640, 25920, 43200, 86400]) 
time_bins = [0, 720, 2160, 4320, 8640, 25920, 43200, 86400]

# time_points = np.array([6, 12, 24, 48, 72, 168, 336, 672]) 
# time_bins = [0, 6, 12, 24, 48, 72, 168, 336, 672]

# time_points = np.array([24, 72, 168, 336, 504])
# time_bins = [0, 24, 72, 168, 336, 504]

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# save checkpoint dir and info dir
os.makedirs(save_ckpt_dir, exist_ok=True)
os.makedirs(save_info_dir, exist_ok=True)

# Create Logger
writer = SummaryWriter(save_info_dir)
    
class SurvivalPredictModel(nn.Module):

    def __init__(self, text_dim=768, text_segment=3, tabnet_dim=512, hidden_dim=512, num_label_bins=8):

        super(SurvivalPredictModel, self).__init__()

        self.ecg_model = build_model_from_checkpoint(checkpoint_path=(os.path.join(ecg_base_checkpoint_path))).to(device)
        self.text_model = T5EncoderModel.from_pretrained(t5_checkpoint_path)
        self.tabnet_model = TabNetWithEmbed()

        # load checkpoint
        ecg_model_weights = torch.load(ecg_pretrained_checkpoint_path, weights_only=True)
        ecg_state_dict = {k.replace('pretrained_model.', ''): v for k, v in ecg_model_weights.items()}

        missing_keys_ecg = self.ecg_model.load_state_dict(ecg_state_dict, strict=False)
        missing_keys_tabnet = self.tabnet_model.load_state_dict(torch.load(tabnet_checkpoint_path, weights_only=True), strict=False)
        print(f"\nThese weights cannot be loaded:\n ecg_keys: {missing_keys_ecg} \n tabnet keys: {missing_keys_tabnet}\n")

        # whether freeze ecg model and Tabnet
        if FREEZE_PRETRAINED_MODEL:
            for model in [self.ecg_model, self.text_model, self.tabnet_model]:
                for param in model.parameters():
                    param.requires_grad = False
        else:
            for param in self.text_model.parameters():
                param.requires_grad = False      

        self.text_dim = text_segment * text_dim
        self.ecg_dim = 2 * self.ecg_model.proj.in_features
        self.tabnet_dim = tabnet_dim
        self.hidden_dim = hidden_dim
    
        self.ecg_proj = nn.Sequential(
            nn.Linear(self.ecg_dim, self.hidden_dim),
            nn.SiLU()
        )

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU()
        )

        self.tabnet_proj = nn.Sequential(
            nn.Linear(self.tabnet_dim, self.hidden_dim),
            nn.SiLU()
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, num_label_bins)
        )

        for proj in [self.ecg_proj, self.text_proj, self.tabnet_proj, self.fusion_proj]:
            for layer in proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
        print("Model proj layer init successfully.")

    def get_ecg_features(self, x):
        res = self.ecg_model(source=x)
        return res["encoder_out"].mean(dim=1)

    def forward(self, ecg_segment_0, ecg_segment_1, text, attention_mask, tabnet_input):
        
        # ECG Model
        ecg_feature_0 = self.get_ecg_features(ecg_segment_0)
        ecg_feature_1 = self.get_ecg_features(ecg_segment_1)

        # 2 * (bs, 768) -> (bs, 1536) -> (bs, 512)
        combined_ecg_feature = torch.cat([ecg_feature_0, ecg_feature_1], dim=-1)
        ecg_embedding = self.ecg_proj(combined_ecg_feature)

        # (bs, 3, 512) -> (bs*3, 512)
        batch_size = text.size(0)
        flat_inputs = text.view(-1, text.size(-1))
        flat_attention = attention_mask.view(-1, attention_mask.size(-1))

        # encode all segments
        outputs = self.text_model(input_ids=flat_inputs, attention_mask=flat_attention)

        # (bs * text_segment, 768) -> (bs, 768 * text_segment) = (bs, 2304)
        text_feature = outputs.last_hidden_state[:, 0, :]
        text_feature = text_feature.reshape(batch_size, -1)

        # (bs, 768 * text_segment) -> (bs, 512)
        text_embedding = self.text_proj(text_feature)

        # Tabnet
        # tabnet_embedding = self.tabnet_model(tabnet_input)
        tabnet_feature = self.tabnet_model(tabnet_input)
        tabnet_embedding = self.tabnet_proj(tabnet_feature)

        # 3 * (bs, 512) -> (bs, 512*3) -> (bs, 8)
        combined_embedding = torch.cat([ecg_embedding, text_embedding, tabnet_embedding], dim=-1)
        logits = self.fusion_proj(combined_embedding)
    
        return logits


text_tokenizer = T5Tokenizer.from_pretrained(
    t5_checkpoint_path,
    legacy=False,           # 启用新tokenizer行为
    use_fast=True,          # 使用Rust实现的快速tokenizer
    truncation_side="right" # 与T5预训练设置保持一致
)
fusion_model = SurvivalPredictModel(text_dim=768, text_segment=3, tabnet_dim=512, hidden_dim=hidden_dim, num_label_bins=len(time_points)).to(device)

train_dataset = SurvivalPredictDataset(
    split="train",
    records_path=records_path,
    data_dir=data_dir,
    ecg_sample_rate=500,
    ecg_target_size=2500,
    text_tokenizer=text_tokenizer,
    text_max_length=512,
    text_max_segment=3,
    time_bins=time_bins
)

val_dataset = SurvivalPredictDataset(
    split="valid",
    records_path=records_path,
    data_dir=data_dir,
    ecg_sample_rate=500,
    ecg_target_size=2500,
    text_tokenizer=text_tokenizer,
    text_max_length=512,
    text_max_segment=3,
    time_bins=time_bins
)

# Create TensorDataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=8, shuffle=False, drop_last=True)

trainable_parameters = []
count_trainable_parameters = 0

print("\nTrainable Parameters: \n")
for name, param in fusion_model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.numel()}")
        trainable_parameters.append(param)
        count_trainable_parameters += param.numel()

print(f"\nTrainable parameters total count: {count_trainable_parameters}\n")
optimizer = optim.Adam(trainable_parameters, lr=learning_rate, betas=betas, weight_decay=weight_decay)

# criterion = NLLLogistiHazardLoss()
criterion = WeightedNLLLogistiHazardLoss(num_bins=len(time_points), alpha=0.5)

best_val_loss = float('inf')  # Initialize best validation loss
best_val_tdauc = 0.0
best_val_acc = 0.0
best_val_auc = 0.0

train_metrics = {'loss': [], 'tdAUC': [], 'acc': [], 'auc': []}
val_metrics = {'loss': [], 'tdAUC': [], 'acc': [], 'auc': []}


def save_checkpoint(local, best, type, epoch):
    if (type == 'loss' and local <= best) or (type != 'loss' and local >= best):
        best = local
        for old_file in glob.glob(os.path.join(save_ckpt_dir, f'best-{type}-*.pth')):
            os.remove(old_file)
        save_path = os.path.join(save_ckpt_dir, f'best-{type}-{local:.2f}-epoch-{epoch}.pth')
        torch.save(fusion_model.state_dict(), save_path)
        print(f"Best {type} model weights saved at '{save_path}', val {type} is {local}")
    return best


def compute_window_accuracy(y_probs, y_event_idx, y_events, num_window):

    auc_list = []
    acc_list = []
    for timepoint_id in range(num_window):
        
        if timepoint_id != num_window - 1:
            mask = ( y_event_idx <= timepoint_id)  # 包括删失样本（需处理）
            y_true = (y_events[mask] == 1).astype(int)
            y_score = 1 - y_probs[mask, y_event_idx[mask]]
        else:
            y_true = y_events.astype(int)
            y_score = 1 - y_probs[:, -1]

        if len(np.unique(y_true)) >= 2:  # 确保有正负样本
            
            auc_list.append(roc_auc_score(y_true, y_score))
            threshold_youden, _ = find_optimal_threshold(y_true, y_score, method='youden')
            y_pred = (y_score >= threshold_youden)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            acc = (tp + tn) / (tp + tn + fp + fn)
            acc_list.append(acc)

    return auc_list, acc_list


# Training loop
for epoch in range(num_epochs):
    
    fusion_model.train()
    running_loss = 0.0
    surv_prob_train, events_train, durations_train, delta_hours_train = [], [], [], []

    print(f"\nStart Training! Now is epoch {epoch}")
    for batch in tqdm(train_loader, desc=f'Train Epoch {epoch}'):

        segment_0 = batch["segment_0"]
        segment_1 = batch["segment_1"]
        train_idx = batch['duration_idx']
        events = batch['event']
        delta_hours = batch['delta_hours']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        tabnet_data = batch['tabnet_data']

        # [bs, 12, 2500], [bs, 12, 2500], [bs], [bs]
        segment_0, segment_1, train_idx, events,  = segment_0.to(device), segment_1.to(device), train_idx.to(device), events.to(device)
        # [bs, 3, 512], [bs, 3, 512], [bs, 17]
        input_ids, attention_mask, tabnet_data = input_ids.to(device), attention_mask.to(device), tabnet_data.to(device)

        optimizer.zero_grad()
        outputs = fusion_model(ecg_segment_0=segment_0, ecg_segment_1=segment_1, text=input_ids, attention_mask=attention_mask, tabnet_input=tabnet_data)
        logits = outputs.squeeze()

        loss = criterion(logits, train_idx, events)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 收集评估数据
        with torch.no_grad():
            surv_prob = torch.cumprod(1 - torch.sigmoid(logits), dim=1)
            surv_prob_train.append(surv_prob.cpu())
            events_train.append(batch['event'])
            durations_train.append(batch['duration_idx'])
            delta_hours_train.append(batch['delta_hours'])
        
    y_surv_probs = torch.cat(surv_prob_train).numpy()
    y_events_train = torch.cat(events_train).numpy()
    y_event_idx_train = torch.cat(durations_train).numpy()
    y_delta_hours_train = torch.cat(delta_hours_train).numpy()

    epoch_loss = running_loss / len(train_loader)
    train_metrics['loss'].append(epoch_loss)

    surv_df_train = pd.DataFrame(y_surv_probs, columns=time_points)
    evaluator_train = EvalSurv(surv_df_train.T, y_delta_hours_train, y_events_train, censor_surv='km')

    # 计算时间依赖性AUC (tdAUC)和准确率
    train_auc = evaluator_train.concordance_td('antolini')
    train_metrics['tdAUC'].append(train_auc)
    train_auc_list, train_acc_list = compute_window_accuracy(y_surv_probs, y_event_idx_train, y_events_train, len(time_points))
    train_metrics['auc'].append(np.mean(train_auc_list))
    train_metrics['acc'].append(np.mean(train_acc_list))

    writer.add_scalar('train/Loss', train_metrics['loss'][-1], epoch)
    writer.add_scalar('train/tdAUC', train_metrics['tdAUC'][-1], epoch)
    writer.add_scalar('train/auc', train_metrics['auc'][-1].mean(), epoch)
    writer.add_scalar('train/acc', train_metrics['acc'][-1].mean(), epoch)
    formatted_auc_list_train = [f"{x:.4f}" for x in train_auc_list]
    formatted_acc_list_train = [f"{x:.4f}" for x in train_acc_list]
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train tdAUC: {train_auc:.4f}, \nTrain Accuracy: {formatted_acc_list_train}, \nTrain AUC: {formatted_auc_list_train}')

    # Validation loop
    print(f"Start Validation! Now is epoch {epoch}")
    fusion_model.eval()
    val_running_loss = 0.0
    surv_prob_val, events_val, durations_val, delta_hours_val = [], [], [], []
    
    with torch.no_grad():
        for val_batch in tqdm(val_loader):

            val_segment_0 = val_batch["segment_0"]
            val_segment_1 = val_batch["segment_1"]
            val_idx = val_batch['duration_idx']
            val_events = val_batch['event']
            val_delta_hours = val_batch['delta_hours']
            val_input_ids = val_batch['input_ids']
            val_attention_mask = batch['attention_mask']
            val_tabnet_data = batch['tabnet_data']

            val_segment_0, val_segment_1, val_idx, val_events = val_segment_0.to(device), val_segment_1.to(device), val_idx.to(device), val_events.to(device)
            val_input_ids, val_attention_mask, val_tabnet_data = val_input_ids.to(device), val_attention_mask.to(device), val_tabnet_data.to(device)

            val_outputs = fusion_model(ecg_segment_0=val_segment_0, ecg_segment_1=val_segment_1, text=val_input_ids, attention_mask=val_attention_mask, tabnet_input=val_tabnet_data)
            val_logits = val_outputs.squeeze()
            val_loss_value = criterion(val_logits, val_idx, val_events)
            val_running_loss += val_loss_value.item()

            surv_prob = torch.cumprod(1 - torch.sigmoid(val_logits), dim=1)
            surv_prob_val.append(surv_prob.cpu())
            events_val.append(val_batch['event'])
            durations_val.append(val_batch['duration_idx'])
            delta_hours_val.append(val_batch['delta_hours'])
        
        y_surv_probs = torch.cat(surv_prob_val).numpy()
        y_events_val = torch.cat(events_val).numpy()
        y_event_idx_val = torch.cat(durations_val).numpy()
        y_delta_hours_val = torch.cat(delta_hours_val).numpy()

    epoch_val_loss = val_running_loss / len(val_loader)
    val_metrics['loss'].append(epoch_val_loss)

    surv_df_val = pd.DataFrame(y_surv_probs, columns=time_points)
    evaluator_val = EvalSurv(surv_df_val.T, y_delta_hours_val, y_events_val, censor_surv='km')

     # 计算时间依赖性AUC (tdAUC)和准确率
    val_auc = evaluator_val.concordance_td('antolini')
    val_metrics['tdAUC'].append(val_auc)
    val_auc_list, val_acc_list = compute_window_accuracy(y_surv_probs, y_event_idx_val, y_events_val, len(time_points))
    val_metrics['auc'].append(np.mean(val_auc_list))
    val_metrics['acc'].append(np.mean(val_acc_list))

    writer.add_scalar('val/Loss', val_metrics['loss'][-1], epoch)
    writer.add_scalar('val/tdAUC', val_metrics['auc'][-1], epoch)
    writer.add_scalar('val/acc', val_metrics['acc'][-1].mean(), epoch)
    writer.add_scalar('val/auc', val_metrics['auc'][-1].mean(), epoch)
    formatted_acc_list_val = [f"{x:.4f}" for x in val_acc_list]
    formatted_auc_list_val = [f"{x:.4f}" for x in val_auc_list]
    print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {epoch_val_loss:.4f}, Val tdAUC: {val_auc:.4f}, \nVal Accuracy: {formatted_acc_list_val}, \nVal AUC: {formatted_auc_list_val}')

    best_val_loss = save_checkpoint(val_metrics['loss'][-1], best_val_loss, 'loss', epoch)
    best_val_tdauc = save_checkpoint(val_metrics['tdAUC'][-1], best_val_tdauc, 'tdauc', epoch)
    best_val_acc = save_checkpoint(val_metrics['acc'][-1], best_val_acc, 'acc', epoch)
    best_val_auc = save_checkpoint(val_metrics['auc'][-1], best_val_auc, 'auc', epoch)

save_path = os.path.join(save_ckpt_dir, f'epoch-final.pth')
torch.save(fusion_model.state_dict(), save_path)
print(f"Final model weights saved at '{save_path}'.")

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, num_epochs + 1), train_metrics['loss'], label='Train Loss')
plt.plot(np.arange(1, num_epochs + 1), val_metrics['loss'], label='Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(os.path.join(save_info_dir, 'loss.png'))

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, num_epochs + 1), train_metrics['acc'], label='Training Accuracy', color='blue')
plt.plot(np.arange(1, num_epochs + 1), val_metrics['acc'], label='Validation Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig(os.path.join(save_info_dir, 'accuracy.png'))

# Plot AUC Curve
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, num_epochs + 1), train_metrics['auc'], label='Training AUC', color='blue')
plt.plot(np.arange(1, num_epochs + 1), val_metrics['auc'], label='Validation AUC', color='orange')
plt.xlabel('Epoch')
plt.ylabel('AUROC')
plt.title('Training and Validation AUROC')
plt.legend()
plt.savefig(os.path.join(save_info_dir, 'auc.png'))

# Plot tdAUC Curve
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, num_epochs + 1), train_metrics['tdAUC'], label='Training tdAUC', color='blue')
plt.plot(np.arange(1, num_epochs + 1), val_metrics['tdAUC'], label='Validation tdAUC', color='orange')
plt.xlabel('Epoch')
plt.ylabel('tdAUC')
plt.title('Training and Validation tdAUC')
plt.legend()
plt.savefig(os.path.join(save_info_dir, 'tdAUC.png'))
