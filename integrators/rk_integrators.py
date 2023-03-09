from .integrator_base import Integrator
import torch
import numpy as np



class RK(Integrator):

    def __init__(self,A,b,d = None,order=0):
        self.A = A
        self.b = b
        self.s = A.shape[0]
        self.order = order
        self.set_torch_params()
        self.d = None
        if type(d).__name__ == "ndarray":
            self.d = d
            self.d_t = torch.from_numpy(self.d).T

    def set_torch_params(self):
        self.A_t = torch.from_numpy(self.A)
        self.b_t = torch.from_numpy(self.b).T
        self.c_t = torch.sum(self.A_t,dim=1)

    def J_t(self,y):
        d = y.shape[-1]//2
        y_q,y_p = torch.split(y,d,dim=-1)
        Jy = torch.cat([y_p,-y_q],axis=-1)
        return Jy

    def f_hat(self,y0,y1 = None,hamiltonian=None,h=None):

        def f(y):
            return hamiltonian.time_derivative(y)
            
        tol = 1e-15

        n = y0.shape[-1]
        if type(y1).__name__ != "bool":
            z_n = torch.concat([h*self.c_t[i]*f(y0) for i in range(self.s)],dim=-1)
        else:
            z_n = torch.concat([h*self.c_t[i]*f(y0) for i in range(self.s)],dim=-1)
        err = torch.norm(y0)
        c = 0

        A_kron_I = torch.kron(self.A_t,torch.eye(y0.shape[-1])).T

        while err > tol and c < 30:
            F = torch.concat([f(y0 + z_n[...,i*n:(i+1)*n]) for i in range(self.s)],dim=-1)
            z_nn = h*F@A_kron_I
            c += 1
            err = torch.norm(z_n - z_nn)
            z_n = z_nn

        if type(self.d).__name__ == "ndarray":
            z_n_unflat = torch.stack([z_n[...,i*n:(i+1)*n] for i in range(self.s)],dim=-1)
            f_hat = (1/h)*(z_n_unflat@self.d_t).squeeze(-1)
        else:
            F = torch.stack([f(y0 + z_n[...,i*n:(i+1)*n]) for i in range(self.s)],dim=-1)
            f_hat = (F@self.b_t).squeeze(-1)


        return f_hat

    def integrator(self,y0,y1,hamiltonian,h):
        return y0 + h*self.f_hat(y0,y1,hamiltonian,h)

    def naive_fp_step(self, y0, dt, hamiltonian, tol=1e-10):
        return self.integrator(y0,None,hamiltonian,dt)

    def integrate(self,y0,dt,hamiltonian,n_steps,method = "fp",numpy_out = False):
        ys = torch.zeros([y0.shape[0]] + [n_steps] + list(y0.shape[1:]) )
        ys[:,0] = y0
        yn = y0

        for i in range(1,n_steps):
            yn = self.integrator(yn,yn,hamiltonian,dt)

            if dt > 0:
                ys[:,i] = yn
            else:
                ys[:,n_steps - i] = yn

        if not numpy_out:
            return ys
        else:
            return ys.detach().numpy().T


        
