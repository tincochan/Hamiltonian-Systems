


from scipy.integrate import solve_ivp
import numpy as np
import torch



def get_ys(hamiltonian_system,dt,n_train,y0,sigma):
    ts = np.linspace(0,dt*n_train,n_train+1)
    f = hamiltonian_system.time_derivative
    ft = lambda t,y:f(y)
    ys = solve_ivp(ft,(0,ts[-1]),y0,t_eval=ts,method='DOP853').y

    ys = ys + np.random.normal(0,size=ys.shape)*sigma
    return torch.tensor(ys,requires_grad=True)



def mean_relative_error(y_true, y_pred):
    return  np.mean( np.linalg.norm(y_true - y_pred,axis=0)/np.linalg.norm(y_true,axis=0))
    

def test_network(neural_net,hamiltonian_system,y0,n_fine,n_train,dt,s_ext=1):

    f = hamiltonian_system.time_derivative
    d = hamiltonian_system.d
    ft = lambda t,y:f(y)
    ts = np.linspace(0,dt*n_train,n_train+1)
    ts_fine = np.linspace(0,s_ext*ts[-1],int(s_ext*n_fine*n_train)+1)

    ft_nn = lambda t,y :neural_net.time_derivative(torch.tensor(y.reshape(1,d),requires_grad=True)).detach().numpy()
    
    ys = solve_ivp(ft,(0,ts[-1]),y0,t_eval=ts,method='DOP853').y
    ys_fine = solve_ivp(ft,(0,ts_fine[-1]),y0,t_eval=ts_fine,method='DOP853').y

    diff = hamiltonian_system.hamiltonian(ys_fine) - neural_net.hamiltonian(torch.tensor(ys_fine).T).detach().numpy().T
    hamiltonian_error = np.mean(np.abs(diff - np.mean(diff)))

    ys_nn = solve_ivp(ft_nn,(0,ts[-1]),y0,t_eval=ts,method='DOP853').y
    ys_nn_fine = solve_ivp(ft_nn,(0,ts_fine[-1]),y0,t_eval=ts_fine,method='DOP853').y

    ext_start_idx = int(ts.shape[0])
    if s_ext > 1:
        ext_start_idx = int((s_ext-1)*n_fine*n_train)+1
        e_fine_ext = mean_relative_error(ys_fine[:,ext_start_idx:],ys_nn_fine[:,ext_start_idx:])


    e_fine_int = mean_relative_error(ys_fine[:,:ext_start_idx],ys_nn_fine[:,:ext_start_idx])
    e_int = mean_relative_error(ys,ys_nn)

    result_dict = {
        'ys':{},
        'error':{}
    }

    result_dict['ys'] = {
        'ys':ys,
        'ys_fine':ys_fine,
        'ys_nn_fine':ys_nn_fine,
        'ys_nn':ys_nn,
    }
    result_dict['error'] = {
        r'Interpolation error $e^{i}(\tilde y)$':e_fine_int,
        r'Extrapolation error $e^{e}(\tilde y)$':e_fine_ext,
        r'Hamiltonian error $e(H_{\theta})$':hamiltonian_error,
        'Test error':e_int,
    }

    return result_dict

