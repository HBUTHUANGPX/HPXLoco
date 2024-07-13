import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class HIMEstimator(nn.Module):
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[128, 64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(HIMEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps   = temporal_steps      # 时间步数
        self.num_one_step_obs = num_one_step_obs    # 单个时间步的观察值数
        self.num_latent       = enc_hidden_dims[-1] # 编码器最后一层的隐藏单元数量
        self.max_grad_norm    = max_grad_norm       # 梯度裁剪的阈值
        self.temperature      = temperature         # 在分布计算中使用的温度参数

        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs # 编码器的输入维度，时间步数乘以单个时间步的观察值数量，用于将历史观察数据转换为特征向量
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):                                    # 循环构建多层的全连接神经网络（即多层感知机）
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation] # 在每一层之后都使用了激活函数activation
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]            # 最后一层的输出维度为enc_hidden_dims[-1] + 3
                                                                                     # 表示最终输出的特征向量长度为enc_hidden_dims[-1] + 3。
                                                                                     # 这可能对应着一些特定的特征提取需求，比如将编码器的输出分为速度（3个值）和其他特征（enc_hidden_dims[-1]个值）？？？？
        self.encoder = nn.Sequential(*enc_layers)                   # 定义好的神经网络层组合成一个整体的编码器网络结构

        # Target
        tar_input_dim = self.num_one_step_obs # 单个时间步的观察值数作为目标网络的输入维度
        tar_layers = []
        for l in range(len(tar_hidden_dims)):                                        # 通过一个循环构建了多层的全连接神经网络（多层感知机）
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation] # 在每一层之后都使用了激活函数activation
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])] #最后一层的输出维度为enc_hidden_dims[-1]
        self.target = nn.Sequential(*tar_layers) #

        # Prototype
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1]) #嵌入层
        #大小为num_prototype×enc_hidden_dims[-1]，这表示嵌入矩阵的行数为num_prototype，每行的维度为enc_hidden_dims[-1]。
        #嵌入层通常用于将离散的索引映射为密集的向量表示，这里可能表示了一些原型特征的存储和映射。
        # Optimizer
        self.learning_rate = learning_rate # 学习率
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate) # 优化模型的参数

    def get_latent(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def encode(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None: # 如果提供了lr参数
            self.learning_rate = lr # 更新模型的学习率为新的值
            for param_group in self.optimizer.param_groups: # 并通过遍历优化器的参数组来更新每个参数组的学习率。
                param_group['lr'] = self.learning_rate
                
        vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3].detach() 
        next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3]

        z_s = self.encoder(obs_history) # 从obs_history中使用编码器self.encoder提取特征z_s
        z_t = self.target(next_obs)     # 从next_obs中使用目标网络self.target提取特征z_t
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:] # 提取预测的速度pred_vel

        z_s = F.normalize(z_s, dim=-1, p=2) # 对提取的特征z_s和z_t进行L2归一化
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad(): # 对原型权重进行L2归一化。
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad(): # 通过Sinkhorn算法计算得分score_s和score_t
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1) # 使用计算得到的得分和温度参数计算log_p_s和log_p_t
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean() # 计算交换损失swap_loss和估计损失estimation_loss，并将两者相加得到总损失losses
        estimation_loss = F.mse_loss(pred_vel, vel)
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad() # 将模型的梯度置零
        losses.backward()          # 进行反向传播计算梯度
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm) # 对梯度进行裁剪
        self.optimizer.step()      # 通过优化器更新模型参数。

        return estimation_loss.item(), swap_loss.item() # 返回估计损失和交换损失的数值表示


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None