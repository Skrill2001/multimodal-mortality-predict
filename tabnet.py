import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Sparsemax(nn.Module):
    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        device = input.device  # 动态获取输入设备
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]
        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)
        self.output = torch.max(torch.zeros_like(input), input - taus)
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)
        return output

    def backward(self, grad_output):
        dim = 1
        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))
        return self.grad_input


def initialize_non_glu(module, inp_dim, out_dim):
    gain = np.sqrt((inp_dim + out_dim) / np.sqrt(4 * inp_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain)


class GBN(nn.Module):
    def __init__(self, inp, vbs=128, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs

    def forward(self, x):
        chunk = torch.chunk(x, max(1, x.size(0) // self.vbs), 0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res, 0)


class GLU(nn.Module):
    def __init__(self, inp_dim, out_dim, fc=None, vbs=128):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim, out_dim * 2)
        self.bn = GBN(out_dim * 2, vbs=vbs)
        self.od = out_dim

    def forward(self, x):
        x = self.bn(self.fc(x))
        return x[:, :self.od] * torch.sigmoid(x[:, self.od:])


class FeatureTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs=128):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first = False
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp, out_dim, vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim, out_dim, vbs=vbs))
        # 动态设备感知：在forward中初始化scale
        self.register_buffer('scale', torch.sqrt(torch.tensor([.5])))  # 使用register_buffer

    def forward(self, x):
        # 确保scale和x在同一个设备上
        if not hasattr(self, 'scale'):
            self.scale = torch.sqrt(torch.tensor([.5], device=x.device))  # 动态初始化
        elif self.scale.device != x.device:
            self.scale = self.scale.to(x.device)  # 动态调整设备

        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x


class AttentionTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, relax, vbs=128):
        super().__init__()
        self.register_buffer('r', torch.tensor([relax]))
        self.fc = nn.Linear(inp_dim, out_dim)
        self.bn = GBN(out_dim, vbs=vbs)
        #         self.smax = Sparsemax()
        self.r = torch.tensor([relax])

    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = torch.sigmoid(a * priors.to(a.device))
        priors = priors.to(a.device) * (self.r.to(a.device) - mask)
        return mask


class DecisionStep(nn.Module):
    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs=128):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, n_d + n_a, shared, n_ind, vbs)
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)

    def forward(self, x, a, priors):
        mask = self.atten_tran(a, priors)
        loss = ((-1) * mask * torch.log(mask + 1e-10)).mean()
        x = self.fea_tran(x * mask)
        return x, loss


class TabNet(nn.Module):
    def __init__(self, inp_dim, final_out_dim, n_d=64, n_a=64, n_shared=2, n_ind=2, n_steps=5, relax=1.2, vbs=128):
        super().__init__()
        # 初始化时注册所有子模块
        self.bn = nn.BatchNorm1d(inp_dim)

        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for x in range(n_shared - 1):
                self.shared.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a)))
        else:
            self.shared = None
        self.first_step = FeatureTransformer(inp_dim, n_d + n_a, self.shared, n_ind)
        self.steps = nn.ModuleList()
        for x in range(n_steps - 1):
            self.steps.append(DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs))
        self.fc = nn.Linear(n_d, final_out_dim)
        self.n_d = n_d

    def forward(self, x):
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d:]
        loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        # 初始化priors时显式指定设备
        priors = torch.ones(x.shape, device=x.device)
        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:, :self.n_d])
            x_a = x_te[:, self.n_d:]
            loss += l
        return self.fc(out), loss
        # return out, loss


class TabNetWithEmbed(nn.Module):
    def __init__(self, n_d=503, n_a=234, n_shared=8, n_ind=8, n_steps=6, relax=1.6730774833562556, vbs=1414):
        super().__init__()

        # 计算输入维度：所有embedding的输出维度之和 + 连续特征的数量
        self.total_embed_dim = 0  # 将在下面计算

        # 使用ModuleList存储所有的Embedding层
        self.cat_embed = nn.ModuleList([
            nn.Embedding(2, 1024),  # sex: 0,1
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024),
            nn.Embedding(2, 1024)

        ])


        # 计算总的embedding维度
        self.total_embed_dim = sum(embed.embedding_dim for embed in self.cat_embed)

        # 连续特征数量：age和maximum_lesion_diameter
        self.num_continuous = 1

        # 总输入维度 = embedding维度 + 连续特征数量
        inp_dim = self.total_embed_dim + self.num_continuous

        # 定义输出维度（根据您的需求设置）
        final_out_dim = 512  # 根据label的分类数设置

        # 初始化TabNet
        self.tabnet = TabNet(inp_dim, final_out_dim, n_d, n_a, n_shared, n_ind, n_steps, relax, vbs)

    def forward(self, x):
        # 获取设备信息
        device = x.device

        # 分离连续特征和离散特征
        # 假设x的前16列是离散特征，后2列是连续特征（age和maximum_lesion_diameter）
        cat_features = x[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]].long().to(device)  # 5个离散特征
        # cat_features = x[:, [0, 2, 3, 5, 7, 8]].long().to(device)  # 5个离散特征
        cont_features = x[:, [0]].to(device)  # 2个连续特征

        # 处理离散特征
        embeddings = []
        for i, embed_layer in enumerate(self.cat_embed):
            embeddings.append(embed_layer(cat_features[:, i]))

        # 合并所有特征
        cat_features = torch.cat(embeddings, 1)
        x = torch.cat((cat_features, cont_features), 1).contiguous()

        # 通过TabNet
        x, loss = self.tabnet(x)
        # x = F.softmax(x, dim=-1)
        return x
