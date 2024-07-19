import torch
import netron
import torch
from torch import nn
from torchinfo import summary

policy = torch.jit.load("/home/hpx/HPXLoco/livelybot_rl_control/logs/Pai_ppo/exported/policies/him_policy.pt")
# policy  = torch.load('/home/hpx/HPXLoco/livelybot_rl_control/logs/Pai_ppo/Jul14_19-19-06_v1/model_0.pt')

print(policy)
a= [1,2,3,4]
print(a[:2])
