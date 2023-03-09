import torch
from torch import optim

from utils.neural_nets import HNN
from integrators.mirk_integrators import MIRK
from utils.training import train_on_seq
from experiments.testing import get_ys,test_network
from experiments.plotting import plot_result_phase,plot_result_time




def run_experiment(exp_dict,hamiltonian_system,integrator,plot = False,save = False):

    torch.manual_seed(0)
    t_dtype = torch.float64
    torch.set_default_dtype(t_dtype)

    dt = exp_dict['T']/exp_dict['n_train']
    n_train = exp_dict['n_train']
    sigma = exp_dict['sigma']
    y0 = exp_dict['y0']
    epochs = exp_dict['epochs']

    n_fine =  exp_dict['n_fine']
    s_ext =  exp_dict['s_ext']


    hidden_dim =  exp_dict['hidden_dim']

    integrator =  exp_dict['integrator']

    d = hamiltonian_system.d

    integrator_str = integrator
    
    if integrator[0:4] == "mirk" or integrator[0:2] == "rk":
        neural_net = HNN(input_dim=d,t_dtype=t_dtype,hidden_dim = hidden_dim)
        integrator = MIRK(type=integrator)

    optimizer = optim.LBFGS(neural_net.parameters(), history_size=120,tolerance_grad=1e-9,tolerance_change=1e-9,line_search_fn="strong_wolfe")

    ys_train = get_ys(hamiltonian_system,dt,n_train,y0,sigma).T


    train_on_seq(neural_net,optimizer,ys_train,dt,integrator,epochs)

    result_dict = test_network(neural_net,hamiltonian_system,y0,n_fine,n_train,dt,s_ext)

    
    ys_dict = result_dict['ys']
    error_dict = result_dict['error']



    if plot or save:
        plot_result_phase(ys_train,ys_dict['ys_fine'],ys_dict['ys_nn_fine'],d,dt,integrator_str,save)
        plot_result_time(ys_train,ys_dict['ys_fine'],ys_dict['ys_nn_fine'],dt,n_train,n_fine,d,s_ext,integrator_str,save)

    return error_dict,ys_dict


def run_experiments_seq(exp_dict,hamiltonian_system,n_trains,integrators,plot=False,save=False):
    result_dict = {}
    ys_dict = {}
    for integrator_str in integrators:
        result_dict[integrator_str] = {}
        ys_dict[integrator_str] = {'nn':{},'train':{}}
        for n_train in n_trains:
            ys_dict[integrator_str]['train'][n_train] = {}
            ys_dict[integrator_str]['train'][n_train] = {}
            
            exp_dict['n_train'] = n_train
            exp_dict['integrator'] = integrator_str
            print("dt = ", exp_dict['T']/n_train)
            print("N_train = ", exp_dict['n_train'])
            print(integrator_str)

            #integrator = MIRK(type=integrator_str)
            result_dict[integrator_str][n_train],ys_dict_one = run_experiment(exp_dict,hamiltonian_system,integrator_str,plot,save)
            ys_dict[integrator_str]['nn'][n_train] = ys_dict_one['ys_nn_fine']
            ys_dict[integrator_str]['train'][n_train]['ys_train'] = ys_dict_one['ys']
            ys_dict[integrator_str]['train'][n_train]['ys_fine'] = ys_dict_one['ys_fine']
        
    return result_dict,ys_dict