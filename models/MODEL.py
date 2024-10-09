import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    
    def __init__(self, dim, use_bias=True):
        super( Bottleneck, self ).__init__()
        
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, dim, bias=use_bias),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(dim, dim, bias=use_bias),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(dim, dim, bias=use_bias),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        out = self.bottleneck(x) + x
        out = F.relu(out)
        return out

        
class MODEL(nn.Module):
    
    def __init__(self, num_kpt, use_bias=True):
        super( MODEL, self ).__init__()
        
        self.name = 'MODEL'
        
        self.in_dim = 2 * num_kpt
        self.single_dim = 4 * num_kpt
        self.mixed_dim = 2 * self.single_dim
        
        self.Single_A = nn.Sequential(
            nn.Linear(self.in_dim, self.single_dim, bias=use_bias),
            nn.BatchNorm1d(self.single_dim),
            nn.ReLU(inplace=True)
        )
        
        self.Single_B = nn.Sequential(
            nn.Linear(self.in_dim, self.single_dim, bias=use_bias),
            nn.BatchNorm1d(self.single_dim),
            nn.ReLU(inplace=True)
        )
        
        self.Single_Block_A1 = Bottleneck(self.single_dim, use_bias)
        self.Single_Block_A2 = Bottleneck(self.single_dim, use_bias)
        
        self.Single_Block_B1 = Bottleneck(self.single_dim, use_bias)
        self.Single_Block_B2 = Bottleneck(self.single_dim, use_bias)
        
        self.Mixed_Block1 = Bottleneck(self.mixed_dim, use_bias)
        self.Mixed_Block2 = Bottleneck(self.mixed_dim, use_bias)
        self.Mixed_Block3 = Bottleneck(self.mixed_dim, use_bias)
        
        self.fc = nn.Sequential(
            nn.Linear(self.mixed_dim, 3)
        )
        self.P1_fc = nn.Sequential(
            nn.Linear(self.mixed_dim, 12)
        )
        self.P2_fc = nn.Sequential(
            nn.Linear(self.mixed_dim, 12)
        )
    
    def forward(self, x1, x2):
        
        x1 = self.Single_A(x1)
        x1 = self.Single_Block_A1(x1)
        x1 = self.Single_Block_A2(x1)
        
        x2 = self.Single_B(x2)
        x2 = self.Single_Block_B1(x2)
        x2 = self.Single_Block_B2(x2)   

        x = torch.cat( (x1, x2), dim=1 )
        
        x = self.Mixed_Block1(x)
        x = self.Mixed_Block2(x)
        x = self.Mixed_Block3(x)
        
        X = self.fc(x)
        P1_vec = self.P1_fc(x)
        P2_vec = self.P2_fc(x)

        P1 = P1_vec.view(-1, 3, 4)
        P2 = P2_vec.view(-1, 3, 4)
        
        ones = torch.ones(X.shape[0], 1).to(X.device)
        X_homo = torch.cat((X, ones), dim=1).unsqueeze(-1)

        x1_hat = torch.bmm(P1, X_homo).squeeze(-1)
        x2_hat = torch.bmm(P2, X_homo).squeeze(-1)
        
        return X, x1_hat, x2_hat, P1, P2
    
    def __str__(self):
        return self.name
        