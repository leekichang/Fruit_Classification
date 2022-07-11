import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        '''
        *args
        None        
        *return
        None
        '''
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=3*224*224, out_features=128, bias=True),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=6, bias = True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        '''
        *args
        x(tensor) B C H W
        B: Batch Size
        C: Channel
        H: Height
        W: Width
        *return
        out(tensor) B N
        B: Batch Size
        N: Number of Class
        '''
        x = x.flatten()
        out = self.layer1(x)
        out = self.layer2(out)
        return out

if __name__ == '__main__':
    model = MLP()
    x = torch.randn((1, 3, 224, 224))
    pred = model(x)
    print(pred.shape)