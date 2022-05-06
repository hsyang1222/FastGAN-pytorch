from torch import nn
import torch

class LinBlock(nn.Module) : 
    def __init__(self, infeature=256, outfeature=128) :
        super(LinBlock, self).__init__()
        self.main = nn.Sequential(
            #nn.BatchNorm1d(infeature),
            nn.Linear(infeature,outfeature),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x) : 
        x = self.main(x)
        return x

    
class DirectDiscriminator(nn.Module):        
    def __init__(self) : 
        super(DirectDiscriminator, self).__init__()
        self.bottom = nn.Sequential(
            LinBlock(256,128),
            nn.Linear(128,1),
            #nn.Sigmoid()
        )
    
    def forward(self, x) : 
        x = self.bottom(x)
        return x.view(-1,1)
    
class TransposeDiscriminator(nn.Module):        
    def __init__(self, batch_size=16, feature=256) : 
        super(TransposeDiscriminator, self).__init__()
        self.layer = [nn.Sequential(
            LinBlock(batch_size, batch_size),
            nn.Linear(batch_size, 1),
        ) for i in range(feature)]
        self.layer = torch.nn.ModuleList(self.layer)
    
    def forward(self, x) : 
        result = torch.empty(len(self.layer))
        for index, each_feature in enumerate(x.t()) :
            result[index] = self.layer[index](each_feature)
        return result