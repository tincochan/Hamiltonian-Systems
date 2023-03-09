

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm 
import copy



import matplotlib as mpl


f = 2
def set_plot_params():
    golden_ratio = (5**.5 - 1) / 2
    params = {
        # Use the golden ratio to make plots aesthetically pleasing
        'figure.figsize': [5, 5*golden_ratio],
        # Use LaTeX to write all text
        "text.usetex": True,
        #"font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document, titles slightly larger
        # In the end, most plots will anyhow be shrunk to fit onto a US Letter / DIN A4 page
        "axes.titlesize": 12 + f,
        "axes.labelsize": 12 + f,
        "font.size": 12 + f,
        "legend.fontsize": 12 + f,
        "xtick.labelsize": 12 + f,
        "ytick.labelsize": 12 + f,
    }
    mpl.rcParams.update(params)

set_plot_params()

global save_dir
save_dir = "plots/"


def plot_result_phase(ys_train,ys_fine,ys_hat_nn_fine,d,dt = None,integrator_str = None,save=False):


    if save:
        file_name = "result_phase_" +save + "_" + integrator_str + "_" + str(round(dt,2))

    ys_train = ys_train.detach().numpy().T

    for j in range(d//2):
        plt.rcParams["figure.figsize"] = (5,5)
        plt.title(" ")

        plt.plot(ys_fine[j,:],ys_fine[j+d//2,:],c="tab:blue",label=r"True trajectory $y(t)$")
        plt.plot(ys_hat_nn_fine[j,:],ys_hat_nn_fine[j+d//2,:],linestyle='--',c="tab:red",label=r"Neural network $\tilde y_n$")
        plt.scatter(ys_train[j,:],ys_train[j+d//2,:],c="tab:green",label=r"Training data",s=15)
        plt.xlabel("$q_" + str(j+1)+"$")
        plt.ylabel("$p_" + str(j+1)+"$")
        plt.legend(loc = 2)
        if save:        
            plt.savefig(save_dir + file_name + str(j) +  ".pdf", format='pdf', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        break


def plot_result_time(ys_train,ys_fine,ys_hat_nn_fine,dt,n_train,n_fine,d,s_ext=1,integrator_str = None,save=False):

    if save:
        file_name = "result_time_" + save + "_" + integrator_str + "_" + str(round(dt,2))

    ts = np.linspace(0,dt*n_train,n_train+1)
    ts_fine = np.linspace(0,s_ext*ts[-1],int(s_ext*n_fine*n_train)+1)

    ys_train = ys_train.detach().numpy().T

    plt.rcParams["figure.figsize"] = (6,2*d)
    fig, axs = plt.subplots(d, 1, sharex='col')

    fig.subplots_adjust(top=0.8,wspace=0,hspace=0)

    for j in range(d):
        axs[j].scatter(ts,ys_train[j,:],c="tab:green",label="Training data",s=15)
        axs[j].plot(ts_fine,ys_fine[j,:],c="tab:blue",label=r"True trajectory $y(t)$")#,linestyle="--",alpha=0.2)
        axs[j].plot(ts_fine,ys_hat_nn_fine[j,:],linestyle='--',c="tab:red",label=r"Neural network $\tilde y_n$")
        axs[j].set_ylabel("$y_" + str(j+1)+"$")

    plt.xlabel("$t$")
    axs[j].legend(loc=1)

    if save:        
        plt.savefig(save_dir + file_name + ".pdf", format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def plot_result_time_multiple(ys_dict,exp_dict,n_trains,n_step_size=0,n_dim=0,save=False):


    n_train = n_trains[n_step_size]
    dt = exp_dict['T']/n_train


    n_fine =  exp_dict['n_fine']
    s_ext =  exp_dict['s_ext']

    n_plots = len(ys_dict.keys())

    if save:

        file_name = "result_time_multiple" + "_" +  save

    ts = np.linspace(0,dt*n_train,n_train+1)
    ts_fine = np.linspace(0,s_ext*ts[-1],int(s_ext*n_fine*n_train)+1)



    plt.rcParams["figure.figsize"] = (6,1.3*n_plots)
    fig, axs = plt.subplots(n_plots, 1, sharex='col')

    fig.subplots_adjust(top=0.8,wspace=0.1,hspace=0)

    l = n_dim #Select dim, max d
    k = n_step_size #Select step size

    j = 0
    for integrator_str in ys_dict:
        i = 0
        for n_train in ys_dict[integrator_str]['nn']:
            if i == k:
                if j == 0:
                    axs[j].scatter(ts,ys_dict[integrator_str]['train'][n_train]['ys_train'][l,:],c="tab:green",label="Training data",s=15)
                    axs[j].plot(ts_fine,ys_dict[integrator_str]['train'][n_train]['ys_fine'][l,:],c="tab:blue",label=r"True trajectory $y(t)$")#,linestyle="--",alpha=0.2)
                    axs[j].plot(ts_fine,ys_dict[integrator_str]['nn'][n_train][l,:] ,linestyle='--',c="tab:red",label=r"Neural network $\tilde y_n$")
                else:
                    axs[j].scatter(ts,ys_dict[integrator_str]['train'][n_train]['ys_train'][l,:],c="tab:green",s=15)
                    axs[j].plot(ts_fine,ys_dict[integrator_str]['train'][n_train]['ys_fine'][l,:],c="tab:blue")#,linestyle="--",alpha=0.2)
                    axs[j].plot(ts_fine,ys_dict[integrator_str]['nn'][n_train][l,:] ,linestyle='--',c="tab:red")

                axs[j].set_ylabel("$y_" + str(l+1)+"$")
                axs[j].legend([],[],title = integrator_str.upper(),loc=1)

            i += 1

        j += 1

    plt.xlabel("$t$")
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05),
            fancybox=True, shadow=True,ncol=3)

    if save:        
        plt.savefig(save_dir + file_name + ".pdf", format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()




def plot_convergence(result_dict,exp_dict,n_trains,save=False):

    if save:
        file_name = "error_" + save


    x = np.linspace(0.0, 1.0, 9)
    x = np.concatenate([x[0:4],[x[6]],[x[8]]])
    rgb = cm.get_cmap("tab10")(x)

    step_sizes = exp_dict['T']/n_trains

    conv_res = copy.deepcopy(result_dict)

    for integrator_str in result_dict:
        for n_train in result_dict[integrator_str]:
            for error_metric in result_dict[integrator_str][n_train]:
                conv_res[integrator_str][error_metric] = []
                
    for integrator_str in result_dict:
        for n_train in result_dict[integrator_str]:
            for error_metric in result_dict[integrator_str][n_train]: 
                error = result_dict[integrator_str][n_train][error_metric]
                conv_res[integrator_str][error_metric].append(error)
            
    n_ticks = len(conv_res[integrator_str][error_metric])
    plt.rcParams["figure.figsize"] = (6,2*3)

    fig, axs = plt.subplots(3,1, sharex='col')
    fig.subplots_adjust(top=0.8,wspace=0,hspace=0)

    n_integrators = len(result_dict.keys())
    if n_integrators%2 == 0:
        ncol = n_integrators//2
    else:
        ncol = n_integrators//2 + 1


    k = 0
    for error_metric in result_dict[integrator_str][n_train]:
        if error_metric != "Test error":
            xs = []
            axs[k].set_yscale("log")
            axs[k].legend([],[],title = error_metric,loc=1)

            i = 0
            for integrator_str in result_dict:
                j = 0
                for e in conv_res[integrator_str][error_metric]:
                    x = [0.25*(i+j*(n_integrators+1))]
                    xs.append(x[0])
                    if j == 0 and k == 0:
                        axs[k].bar(x,e,color=rgb[i],width=0.15,label=integrator_str.upper())
                    else:
                        axs[k].bar(x,e,color=rgb[i],width=0.15)
                    j+=1
                i += 1
            k+=1

    xs = np.sort(xs)
    ticks = [np.mean(xs[n_integrators*i:n_integrators*(i+1)]) for i in range(n_ticks)]
    xticks = ( r'$h = $ ' + str(round(h,2)) + r', $N = $ ' + str(int(n)) for h,n in zip(step_sizes,n_trains))

    axs[-1].set_xticks(ticks, xticks)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05),
            fancybox=True, shadow=True,ncol=ncol)
    if save:        
        plt.savefig(save_dir + file_name +  ".pdf", format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()