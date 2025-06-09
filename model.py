import torch
import torch.nn as nn
from torch import Tensor
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import numpy as np
import os
from sklearn.metrics import roc_curve, f1_score, precision_recall_curve

class FineTunedWav2Vec2Model(nn.Module):
    def __init__(self, pretrained_model, middle_dim, dropout, FREEZE_PRETRAINED_MODEL, MULTI_DEVICES):

        super(FineTunedWav2Vec2Model, self).__init__()

        self.pretrained_model = pretrained_model
        self.freeze = FREEZE_PRETRAINED_MODEL
        self.multi_devices = MULTI_DEVICES

        if self.freeze:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        self.encoder_embed_dim = self.pretrained_model.proj.in_features
    
        self.proj = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(2 * self.encoder_embed_dim, middle_dim),
            nn.SiLU(),
            nn.Linear(middle_dim, 1)
        )

        if not (self.multi_devices and torch.distributed.get_rank() != 0):
            for layer in self.proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
            print("Model proj layer init successfully.")

    def get_features(self, x):
        if self.freeze:
            with torch.no_grad():
                res = self.pretrained_model(source=x)
        else:
            res = self.pretrained_model(source=x)
        return res["encoder_out"].mean(dim=1)

    def forward(self, segment_0, segment_1):
        
        feature_0 = self.get_features(segment_0)
        feature_1 = self.get_features(segment_1)

        combined_feature = torch.cat([feature_0, feature_1], dim=-1)
        logits = self.proj(combined_feature)
        return logits
    
def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")

def nll_logistic_hazard(phi: Tensor, idx_durations: Tensor, events: Tensor,
                        reduction: str = 'mean') -> Tensor:
    """Negative log-likelihood of the discrete time hazard parametrized model LogisticHazard [1].
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.

    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)
    bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)


def weighted_nll_logistic_hazard(
    phi: Tensor, 
    idx_durations: Tensor, 
    events: Tensor,
    weights_pos: Optional[Tensor] = None,  # 各区间正样本权重 (shape: [num_bins])
    weights_neg: Optional[Tensor] = None,  # 各区间负样本权重
    reduction: str = 'mean'
) -> Tensor:
    
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)
    
    # 如果有权重，就要给权重赋值
    if weights_pos is not None or weights_neg is not None:
        weights = torch.ones_like(phi)
        duration_indices = idx_durations.expand_as(phi)
        if weights_pos is not None:
            weights[y_bce == 1] = weights_pos[duration_indices[y_bce == 1]]
        if weights_neg is not None:
            weights[y_bce == 0] = weights_neg[duration_indices[y_bce == 0]]
    else:
        weights = None
    
    # 带权重的BCE
    bce = F.binary_cross_entropy_with_logits(
        phi, y_bce, 
        weight=weights,
        reduction='none'
    )
    
    # 剩余逻辑不变
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)

class _Loss(torch.nn.Module):
    """Generic loss function.
    
    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction


class NLLLogistiHazardLoss(_Loss):
    """Negative log-likelihood of the hazard parametrization model.
    See `loss.nll_logistic_hazard` for details.
    
    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.
    """
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        return nll_logistic_hazard(phi, idx_durations, events, self.reduction)

class WeightedNLLLogistiHazardLoss(_Loss):
    def __init__(self, num_bins=8, alpha=0.5, reduction='mean'):
        super().__init__(reduction)
        self.num_bins = num_bins
        self.alpha = alpha

    def get_weights(self, idx_durations: Tensor, events: Tensor, device):

        pos_counts = torch.zeros(self.num_bins).to(device)
        neg_counts = torch.zeros(self.num_bins).to(device)
            
        for t in range(self.num_bins):
            mask_t = (idx_durations == t)
            mask_after_t = (idx_durations > t)
            pos_counts[t] += (events[mask_t] == 1).sum()
            neg_counts[t] += (events[mask_t] == 0).sum() + mask_after_t.sum()

        weights_pos = (neg_counts / (pos_counts + 1e-6)) ** self.alpha
        weights_neg = (pos_counts / (neg_counts + 1e-6)) ** self.alpha

        weights_pos = torch.clamp(weights_pos, min=0.1, max=10.0).to(device)
        weights_neg = torch.clamp(weights_neg, min=0.1, max=10.0).to(device)
        
        return weights_pos, weights_neg

    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        
        device = phi.device
        weights_pos, weights_neg = self.get_weights(idx_durations, events, device)
        return weighted_nll_logistic_hazard(
            phi, idx_durations, events,
            weights_pos, weights_neg,
            self.reduction
        )
    

def find_optimal_threshold(y_true, y_prob, method='youden', target_metric=None, target_value=None):
    """
    根据AUC曲线或PR曲线选择最优分类阈值。

    参数:
    -----------
    y_true : array-like
        真实标签（0或1）。
    y_prob : array-like
        预测为正类的概率（形状与y_true相同）。
    method : str, 默认'youden'
        阈值选择方法，可选:
        - 'youden': 最大化Youden's J指数（灵敏度 + 特异度 - 1）
        - 'f1': 最大化F1-score
        - 'target': 根据目标召回率或精确率选择阈值
    target_metric : str, 可选
        当method='target'时指定目标指标，可选'recall'或'precision'。
    target_value : float, 可选
        当method='target'时指定目标值（如0.9表示90%召回率）。

    返回:
    -----------
    float
        最优阈值。
    dict
        包含阈值下对应指标（如F1、召回率、精确率等）。
    """
    if method == 'youden':
        # 计算ROC曲线并最大化Youden's J指数
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        metrics = {
            'threshold': optimal_threshold,
            'recall': tpr[optimal_idx],
            'specificity': 1 - fpr[optimal_idx],
            'youden_index': j_scores[optimal_idx]
        }

    elif method == 'f1':
        # 通过PR曲线最大化F1-score
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        metrics = {
            'threshold': optimal_threshold,
            'precision': precision[optimal_idx],
            'recall': recall[optimal_idx],
            'f1_score': f1_scores[optimal_idx]
        }

    elif method == 'target' and target_metric and target_value is not None:
        # 根据目标召回率或精确率选择阈值
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        if target_metric == 'recall':
            optimal_idx = np.argmin(np.abs(recall - target_value))
        elif target_metric == 'precision':
            optimal_idx = np.argmin(np.abs(precision - target_value))
        else:
            raise ValueError("target_metric必须是'recall'或'precision'")
        optimal_threshold = thresholds[optimal_idx]
        metrics = {
            'threshold': optimal_threshold,
            'precision': precision[optimal_idx],
            'recall': recall[optimal_idx]
        }

    else:
        raise ValueError("无效的method或缺少target参数")

    return optimal_threshold, metrics


def find_checkpoint(directory, epoch=149):
    matched_file_path = ''
    suffix = f'epoch-{epoch}.pth'
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                matched_file_path = os.path.join(root, file)
                break
    if matched_file_path != '':
        matched_file_name = os.path.basename(matched_file_path)
        matched_file_name = os.path.splitext(matched_file_name)[0]
    else:
        raise ValueError(f"Cannot find pth end with {suffix}")
    return matched_file_path, matched_file_name


class FineTunedWav2Vec2Model_full(nn.Module):
    def __init__(self, pretrained_model, middle_dim, dropout, num_label_bins):

        super(FineTunedWav2Vec2Model_full, self).__init__()

        self.pretrained_model = pretrained_model
        self.encoder_embed_dim = self.pretrained_model.proj.in_features
    
        self.proj = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(self.encoder_embed_dim, middle_dim),
            nn.SiLU(),
            # nn.Linear(middle_dim, middle_dim//2),
            # nn.SiLU(),
            nn.Linear(middle_dim, num_label_bins)
        )

        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        print("Model proj layer init successfully.")

    def get_features(self, x):
        res = self.pretrained_model(source=x)
        return res["encoder_out"].mean(dim=1)

    def forward(self, segment_0, segment_1):
        
        ecg_data = torch.cat([segment_0, segment_1], dim=-1)
        res = self.pretrained_model(source=ecg_data)
        ecg_feature = res["encoder_out"].mean(dim=1)
        logits = self.proj(ecg_feature)
        return logits
    

class FineTunedWav2Vec2Model_concat(nn.Module):
    def __init__(self, pretrained_model, middle_dim, dropout, num_label_bins):

        super(FineTunedWav2Vec2Model_concat, self).__init__()

        self.pretrained_model = pretrained_model
        self.encoder_embed_dim = self.pretrained_model.proj.in_features
    
        self.proj = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(self.encoder_embed_dim, middle_dim),
            nn.SiLU(),
            # nn.Linear(middle_dim, middle_dim//2),
            # nn.SiLU(),
            nn.Linear(middle_dim, num_label_bins)
        )

        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        print("Model proj layer init successfully.")
    
    def get_features(self, x):
        res = self.pretrained_model(source=x)
        return res["encoder_out"]

    def forward(self, segment_0, segment_1):
        feature_0 = self.get_features(segment_0)
        feature_1 = self.get_features(segment_1)
        combined_feature = torch.cat([feature_0, feature_1], dim=1)
        combined_feature = combined_feature.mean(dim=1)
        logits = self.proj(combined_feature)
        return logits
