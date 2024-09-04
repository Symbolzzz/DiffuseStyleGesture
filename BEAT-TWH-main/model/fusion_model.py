import torch.nn as nn
import torch

class FeatureFusion(nn.Module):
    def __init__(self, dim_g, dim_f, dim):
        super(FeatureFusion, self).__init__()
        # 定义一维卷积层
        self.conv = nn.Conv1d(in_channels=dim_g + dim_f, out_channels=dim, kernel_size=1)
        
    def forward(self, gesture_features, facial_features):
        # 在特征维度上拼接 [bs, seqlength, dim_g + dim_f]
        concatenated_features = torch.cat((gesture_features, facial_features), dim=-1)
        
        # 将 [bs, seqlength, dim_g + dim_f] 转换为 [bs, dim_g + dim_f, seqlength]
        concatenated_features = concatenated_features.transpose(1, 2)
        
        # 通过卷积层映射到目标维度
        fused_features = self.conv(concatenated_features)
        
        # 再转换回 [bs, seqlength, dim]
        fused_features = fused_features.transpose(1, 2)
        
        return fused_features