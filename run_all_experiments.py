import numpy as np
import torch
from utils.analytical_hamiltonian import *
from experiments.numerical_experiments import run_experiments_seq
from experiments.plotting import plot_convergence, plot_result_time_multiple


torch.manual_seed(0)
t_dtype = torch.float64
torch.set_default_dtype(t_dtype)


#Setting experimental variables

b = 10

exp_dict = {
    'T' :           2*b, #Size of the temporal domain of the training data
    'n_train' :     1*b, #Number of timesteps in the training data
    'sigma' :       0.00, #Standard deviation of the noise (normally distributed perturbations) in the data

    'n_fine':       20, #Number of refinements of the temporal domain to perform interpolation
    's_ext':        4,  #Number of times the temporal domain is extended in testing to perform extrapolation

    'hidden_dim':   100, #Hidden dim of neural network
    'epochs':       100, #Number of epoch to train
}

n_trains = np.array([1,2,4])*b
integrators = ['mirk2','mirk3','rk4','mirk4','mirk5','mirk6']

########### Henon-Héiles ##################

exp_dict['y0'] = np.array([0.2,0.35,-0.3,0.2])
hamiltonian_system = HamiltonianSystem(hamiltonian_henon_heiles)

save_every_exp = False  #Change to "henon" to save phase and time plots for every single exp.
save_aggregate_results = "henon"
result_dict,ys_dict = run_experiments_seq(exp_dict,hamiltonian_system,n_trains,integrators,plot=False,save=save_every_exp)
plot_result_time_multiple(ys_dict,exp_dict,n_trains,n_step_size=0,n_dim=2,save=save_aggregate_results)
plot_convergence(result_dict,exp_dict,n_trains,save=save_aggregate_results)


########### Fermi-Pasta-Ulam-Tsingou ##################

exp_dict['y0'] = np.array([0.2,0.4,-0.3,0.5])
fermi_pasta = get_fermi_pasta_ulam_tsingou(m = 1,omega = 2)


hamiltonian_system = HamiltonianSystem(fermi_pasta)

save_every_exp = False  #Change to "fput" to save phase and time plots for every single exp.
save_aggregate_results = "fput"
result_dict,ys_dict = run_experiments_seq(exp_dict,hamiltonian_system,n_trains,integrators,plot=False,save=save_every_exp)
plot_result_time_multiple(ys_dict,exp_dict,n_trains,n_step_size=1,n_dim=2,save=save_aggregate_results)
plot_convergence(result_dict,exp_dict,n_trains,save=save_aggregate_results)


########### Double-pendulum ##################

exp_dict['y0'] = np.array([-0.1,0.5,-0.3,0.1])
hamiltonian_system = HamiltonianSystem(hamiltonian_double_pendulum)

save_every_exp = False  #Change to "dp" to save phase and time plots for every single exp.
save_aggregate_results = "dp"
result_dict,ys_dict = run_experiments_seq(exp_dict,hamiltonian_system,n_trains,integrators,plot=False,save=save_every_exp)
plot_result_time_multiple(ys_dict,exp_dict,n_trains,n_step_size=0,n_dim=2,save=save_aggregate_results)
plot_convergence(result_dict,exp_dict,n_trains,save=save_aggregate_results)