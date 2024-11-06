import numpy as np
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

def get_theta_2(l1,l2,xw, yw):
    xw2 = np.power(xw,2)+np.power(yw,2)
    c2 = (xw2 - l2**2-l1**2)/(2*l1*l2)
    print(c2)
    theta21 = np.arccos(c2)
    return np.rad2deg(theta21), np.rad2deg(-theta21)

def get_theta_1(l1,l2,xw,yw,theta2):
    #print(theta2)
    temp = (l2*np.sin(theta2))/(l1+l2*np.cos(theta2))
    theta1 = np.arctan(yw/xw) - np.arctan(temp)
    return np.rad2deg(-theta1)

def get_Theta(x,y,l1,l2,theta1,theta2):
    xw,yw = get_global_coords(x,y,l1,l2,theta1,theta2)
    print(xw,yw)
    theta21,_ = get_theta_2(l1,l2,xw,yw)
    #print(theta2)
    theta11= get_theta_1(l1,l2,xw,yw,theta21)
    return theta11, theta21

def get_global_coords(x,y,l1,l2,theta1,theta2):
    xn = l1 * np.cos(np.deg2rad(theta1)) +l2 * np.cos(np.deg2rad(theta1 + theta2)) + x*np.sin(np.deg2rad(theta1+theta2)) + y*np.cos(np.deg2rad(theta1+theta2))
    yn = l1 * np.sin(np.deg2rad(theta1)) + l2 * np.sin(np.deg2rad(theta1 + theta2)) + x*np.cos(np.deg2rad(theta1+theta2)) + y*np.sin(np.deg2rad(theta1+theta2))
    return xn,yn

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True,
    )

def np_to_th(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float32).reshape(n_samples, -1)

def loss( model: torch.nn.Module,X,l1,l2,theta1,theta2):
    xw,yw = get_global_coords(X[:,0],X[:,1],l1,l2,theta1,theta2)
    inputs2=np_to_th(np.array(xw**2+yw**2))
    inputs = np.array([xw,yw]).T
    inputs = np_to_th(inputs).requires_grad_(True)
    X = np_to_th(X).requires_grad_(True)
    out = model(inputs)
    theta_2 = torch.rad2deg(torch.arccos(((inputs[:,0]**2 + inputs[:,1]**2) - l1**2 - l2**2)/(2*l1*l2)))
    dcos, = grad(theta_2,inputs)
    dtheta2, = grad(out,inputs)
    nloss = dtheta2 - dcos
    return torch.mean(nloss**2)

class Net(nn.Module):
    def __init__(
            self,
            input_dims,
            output_dims,
            n_units=100,
            epochs=1000,
            loss1 = nn.MSELoss(),
            loss2 = None,
            lr = 1e-6,
            loss2_weight = 0.4,
            l1 = 6,
            l2 = 5,
            theta1=0,
            theta2=0,
        ) -> None:
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.theta2 = theta2
        self.theta1 = theta1
        self.epochs = epochs
        self.loss1 = loss1
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.n_units = n_units
        self.layers = nn.Sequential(
            nn.Linear(2, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.out = nn.Linear(32, output_dims)

    def forward(self, x):
        #print(x.shape)
        h = self.layers(x)
        out = self.out(h)
        return out
    def fit(self, X, y):
        xw,yw = get_global_coords(X[:,0],X[:,1],self.l1,self.l2,self.theta1,self.theta2)
        #print(np.array(xw**2+yw**2).shape)
        xt = np_to_th(np.array([xw,yw]).T)
        yt = np_to_th(y)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimizer.zero_grad()
            out = self.forward(xt)
            #print(out)
            loss = self.loss1(yt, out)
            #print(loss)
            if self.loss2:
                loss += self.loss2_weight * self.loss2(self, X,self.l1,self.l2,self.theta1,self.theta2) #+ self.loss2_weight
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if ep % (self.epochs//10) == 0:
                print(f"Epoch {ep} Loss {loss.item()}")
        return losses
    def predict(self, X):
        xw,yw = get_global_coords(X[:,0],X[:,1],self.l1,self.l2,self.theta1,self.theta2)
        xt = np_to_th(np.array([xw,yw]).T)
        self.eval()
        out = self.forward(xt)
        return out.detach().numpy()


class CreateModel:
    def __init__(self,xstart,ystart,xend,yend,l1=6,l2=5,theta1=0,theta2=0,epochs=10000,lr=1e-4,loss2_weight=1) -> None:
        self.X = np.linspace(float(xstart),float(xend),100)
        self.Y = np.linspace(float(ystart),float(yend),100)
        eq = functools.partial(get_Theta,l1=l1,l2=l2,theta1=0,theta2=0)
        self.theta_vals = eq(self.X,self.Y)
        self.model = Net(2,1,epochs=epochs,lr=lr,loss2=loss, loss2_weight=loss2_weight,l1=l1,l2=l2,theta1=theta1,theta2=theta2)
        self.xt = np.linspace(xstart,0.5*xend,30)
        self.t = np.linspace(ystart,0.5*yend,30)
        self.T = eq(self.xt,self.t)
    def train(self):
        X_train = np.array([self.xt,self.t]).T
        y_train = np.array(self.T[1]).reshape(-1,1)
        losses = self.model.fit(X_train, y_train)
        return losses
    def save_model(self):
        torch.save(self.model.stat_dict(),"model.pt")
    
    def model_graphs(self, losses):
        plt.plot(losses)
        plt.yscale('log')
        plt.title("Loss during Training")
        plt.ylabel('log losses')
        plt.xlabel('Epochs')
        plt.savefig('loss.png')
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()