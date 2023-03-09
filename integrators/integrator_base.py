
from sympy.matrices import Matrix
from sympy.parsing.latex import parse_latex
import torch
import numpy as np

class Integrator():
    def __init__(self):
        return
    def F(self,y2,y1,dt,hamiltonian):
        if y1.dim() == 1:
            y2 = y2.unsqueeze(0)
            y1 = y1.unsqueeze(0)
            return -y2[0] + self.integrator(y1,y2,hamiltonian,dt)[0]
        else:
            return -y2[0] + self.integrator(y1,y2,hamiltonian,dt)[0]

    def jacobian(self,y2,y1,dt,hamiltonian):
        if len(y2.shape) == 1:
            F = lambda y: self.F(y,y1,dt,hamiltonian)
            return torch.autograd.functional.jacobian(F,y2)
        else:

            jac = []
            for i in range(len(y1)):
                F = lambda y: self.F(y,y1[i],dt,hamiltonian)
                jac.append(torch.autograd.functional.jacobian(F,y2[i] ))
            return torch.stack(jac)

    def newton_step(self,y0,dt,hamiltonian,tol = 1e-16):
        err = torch.norm(self.F(y0,y0,dt,hamiltonian))
        y0 = y0.squeeze(0)
        yn = y0
        c = 0
        A = self.jacobian(yn,y0,dt,hamiltonian)
        while err > tol and c < 5:
            b = self.F(yn,y0,dt,hamiltonian).unsqueeze(len(A.shape)-1)
            yn = yn - torch.linalg.solve(A,b).squeeze(-1)
            err = torch.norm(self.F(yn,y0,dt,hamiltonian))
            c+=1
        return yn



    def fp_step(self,y0,dt,hamiltonian,tol = 1e-16):
        d = y0.shape[-1]//2
        dim = len(y0.shape)
        z_n = torch.zeros_like(y0)
        z_n = dt*hamiltonian.time_derivative(y0)

        err = torch.norm(y0)
        c = 0
        while err > tol and c < 20:
            z_nn = dt*self.f_hat(y0,z_n + y0,hamiltonian,dt)
            err = torch.norm(z_n - z_nn)
            z_n = z_nn
            c += 1

        return z_n + y0

    def fp_step_numpy(self,y0,dt,hamiltonian,tol = 1e-16):
        d = y0.shape[-1]//2
        dim = len(y0.shape)
        z_n = np.zeros_like(y0)
        z_n = dt*hamiltonian.time_derivative(y0)

        err = np.linalg.norm(y0)
        c = 0
        while err > tol and c < 60:
            z_nn = dt*self.f_hat_np(y0,z_n + y0,hamiltonian,dt)
            err = np.linalg.norm(z_n - z_nn)
            z_n = z_nn
            c += 1

        return z_n + y0


    def integrate_torch_fp(self,y0,dt,hamiltonian,n_steps):
        ys = torch.zeros([y0.shape[0]] + [n_steps] + list(y0.shape[1:]) )
        ys[:,0] = y0
        yn = y0
        for i in range(1,n_steps):
            yn = self.fp_step(yn,dt,hamiltonian)
            if dt > 0:
                ys[:,i] = yn
            else:
                ys[:,n_steps - i] = yn
        return ys
    
    def integrate_fp_numpy(self,y0,dt,hamiltonian,n_steps):
        ys = np.zeros([y0.shape[0]] + [n_steps] + list(y0.shape[1:]) )
        ys[:,0] = y0
        yn = y0
        for i in range(1,n_steps):
            yn = self.fp_step_numpy(yn,dt,hamiltonian)
            if dt > 0:
                ys[:,i] = yn
            else:
                ys[:,n_steps - i] = yn
        return ys


    def integrate_torch(self,y0,dt,hamiltonian,n_steps):
        ys = torch.zeros([y0.shape[0]] + [n_steps] + list(y0.shape[1:]) )
        ys[:,0] = y0
        yn = y0
        for i in range(1,n_steps):
            yn = self.newton_step(yn,dt,hamiltonian)
            if dt > 0:
                ys[:,i] = yn
            else:
                ys[:,n_steps - i] = yn
        return ys

    def integrate(self,y0,dt,hamiltonian,n_steps,method = "newton",numpy = False):
        if numpy:
            ys = np.zeros([y0.shape[0]] + [n_steps] + list(y0.shape[1:]) )
        else:
            ys = torch.zeros([y0.shape[0]] + [n_steps] + list(y0.shape[1:]) )

        step_func = None
        if method == "fp_numpy":
            step_func = self.fp_step_numpy
        if method == "newton":
            step_func = self.newton_step
        if method == "fp":
            step_func = self.fp_step

        ys[:,0] = y0
        yn = y0

        for i in range(1,n_steps):
            yn = step_func(yn,dt,hamiltonian)
            if dt > 0:
                ys[:,i] = yn
            else:
                ys[:,n_steps - i] = yn
        return ys




    def integrator(self,y1,y2,neural_net,h):
        raise NotImplementedError

