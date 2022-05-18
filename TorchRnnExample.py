import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.autograd as Variable
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(1)

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(
            input_size = INPUT_SIZE,
            hidden_size = 32,
            num_layers = 1,
            batch_first = True
        )
        self.out = nn.Linear(32,1)
    
    def forward(self,x,h_state):
        r_out, h_state = self.rnn(x,h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:,time_step,:]))
        return torch.stack(outs,dim=1), h_state

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()
h_state = None

plt.figure(1,figsize=(12,5))
plt.ion()

for step in range(120):
    start, end = step*np.pi,(step+1)*np.pi
    steps = np.linspace(start,end,TIME_STEP,dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    
    x = torch.from_numpy(x_np[np.newaxis,:,np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis,:,np.newaxis])
    
    prediction, h_state = rnn(x,h_state)
    h_state = h_state.data
    
    loss = loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    plt.plot(steps,y_np.flatten(),'r-')
    plt.plot(steps,prediction.data.numpy().flatten(),'b-')
    plt.draw()
    plt.pause(0.05)
plt.ioff()
plt.show()        