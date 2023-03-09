from torch import nn
import torch


class HNN(nn.Module):
    def __init__(self,input_dim,t_dtype,hidden_dim):
        super(HNN, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        d = input_dim//2
        I = torch.diag(torch.ones(d))
        J = torch.zeros((input_dim,input_dim))
        J[0:d,d:],J[d:,0:d] = I,-I 
        self.J = J

    def grad(self, x,h=None):
        H = self.seq(x)
        dH = torch.autograd.grad(H,x,create_graph=True,grad_outputs=torch.ones_like(H))[0]
        return dH

    def jac(self,x):
        d = x.shape[-1]
        with torch.enable_grad():
            x = x.requires_grad_(True)
            g = self.time_derivative(x)
            jac_n = [torch.autograd.grad(g[i], (x,), retain_graph=True)[0] for i in range(d)]
            jac = torch.stack(jac_n,1)
            return jac

    def get_L(self,ord=2):
        L=1
        with torch.no_grad():
            for par in self.parameters():
                if len(par.shape) == 2:
                    l = torch.linalg.norm(par.data,ord=ord)
                    L *= l
        return float(L)

    def forward(self,x):
        return self.H(x)

    def time_derivative(self,x):
        d = x.shape[-1]//2
        dh_q,dh_p = torch.split(self.grad(x),d,dim=-1)
        Jdh = torch.cat([dh_p,-dh_q],axis=-1)
        return Jdh

    def H(self,x):
        return self.seq(x)

    def hamiltonian(self,x):
        return self.H(x)






