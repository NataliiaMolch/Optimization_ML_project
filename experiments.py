from hpb import hyperband
from pso import pco
from pbt import pbt_demo
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import os
import sys


from memory_profiler import profile

def params_hyperband(max_epochs):
    """Get the parameters for hyperband and how much time it will take"""
    ## To see the parameters successively halving uncomment the print statements in the method
    R = max_epochs
    eta = 3.
    s_max = int(np.log(R)/np.log(eta))
    B = (s_max + 1) * R

    all_epochs = 0
    all_confs = 0
    for s in range(s_max+1)[::-1]:
        n = int(math.ceil(int(B / R / (s+1)) * pow(eta, s))) 
        r = R * pow(eta, -s)
        n_i = n
        # print('This is for value of S: {}'.format(s))
        for i in range(s+1):
            r_i = r * pow(eta, i)
            all_epochs += n_i * r_i
            all_confs += n_i
            n_i = math.floor(n * pow(eta, -i))
        #     print(i, n_i, r_i)
        # print('\n\n')
    return all_epochs, all_confs


def get_parameters_pco(max_epochs):
    all_epochs, all_confs = params_hyperband(max_epochs)
    n_epochs = all_epochs/all_confs
    n_particles = 5.
    n_iterations = all_confs/n_particles
    return [round(n_iterations), round(n_epochs), round(n_particles)] #args = [n_iterations, n_epochs, n_particles]


def get_parameters_pbt(max_epochs):
    all_epochs, all_confs = params_hyperband(max_epochs)
    n_epochs = all_epochs/all_confs
    process_exp = 5
    parallel_processes = (2 ** (process_exp - 3)) + 1
    steps = (all_confs / process_exp) - 2
    return n_epochs, parallel_processes, steps

"""
max_epochs contain hyperband parameters, basing on which parameters -- args -- for a specific method are computed
"""
@profile(precision=4)
def run_experiment(max_epochs, method = 'hyperband', test_batch_size = 256):
    #compute consistency parameters
    optimizer = None
    args = None
    if method == 'pco':
        args = get_parameters_pco(max_epochs)
        optimizer = pco.PCO(args)
    elif method == 'pbt':
        n_epochs, parallel_processes, steps = get_parameters_pbt(max_epochs)
        optimizer = pbt_demo.PBT(n_epochs, parallel_processes, steps)
    elif method == 'hyperband':
        optimizer = hyperband.HyperBand(max_epochs, test_batch_size)
    else:
        print('ERROR: No such method')
        raise ValueError

    print(f'INFO : parameters of {method} are : {args}\n')
    print('Press any KEY to continue. Press e to exit.')
    a = input()
    if a == 'e':
        return False

    optimizer.search()

    trials, epochs, top_loss, time = optimizer.get_monitored_values()

    with open(f'results/{max_epochs}/' + method + '_experiment.txt', 'w') as f:
        best = optimizer.get_best_conf()

        f.write('Best configuration\t')
        for hp in best[0]:
            f.write(str(hp) + '\t')
        f.write('Best loss\t{}\n'.format(best[1]))

        f.write('Time\tTrials\tEpochs\tTop loss\n')

        for k in range(len(time)):
            f.write('{}\t{}\t{}\t{}\n'.format(time[k], trials[k], epochs[k], top_loss[k]))

    return True
"""
build_plots :
    -methods : list with methods names or method name:  
"""

def build_plots(max_epochs, methods = ['hyperband', 'pco', 'pbt']):

    def get_image_name(methods, max_epochs = max_epochs):
        image_name = f'results/{max_epochs}/'
        for m in methods:
            image_name += m 
            image_name += '_'
        return image_name[:-1] + '.png'


    if type(methods) == str:
        methods = ['methods']

    fig, ax = plt.subplots(3, 1, figsize = (15, 15))
    
    max_ = [-float('inf') for _ in range(4)]
    min_ = [float('inf') for _ in range(4)]

    for method in methods:

        filename = f'results/{max_epochs}/' + method + '_experiment.txt'

        if os.path.exists(filename) == False:
            print('File does not exist. Check the spelling of method {}'.format(method))
            continue

        with open(filename, 'r') as f:
            best_conf_line = f.readline()
            best_loss = f.readline()

            tmp= []

            for line in f:
                tmp.append(line.strip().split('\t'))
        
            tmp = [map(float, t) for t in tmp]
            tmp = np.asarray(list(map(list, zip(*tmp))))

        labels = ['Time', 'Number of configurations cheked', 'Epochs', 'Top loss']

        max_ = [max(max_[i],max(tmp[i])) for i in range(4)]
        min_ = [min(min_[i],min(tmp[i])) for i in range(4)]

        for i in range(3):
            ax[i].plot(tmp[0], tmp[i + 1], label = method)

            ax[i].set_xlabel(labels[0])
            ax[i].set_ylabel(labels[i+1])
            ax[i].grid(True)
            ax[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
            ax[i].yaxis.set_ticks(np.linspace(min_[i+1], max_[i+1], 10, dtype = 'float'))
            if i == 2:
                ax[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%f'))
                ax[i].yaxis.set_ticks(np.linspace(min_[i+1], max_[i+1], 10, dtype = 'float'))
            ax[i].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
            ax[i].xaxis.set_ticks(np.linspace(min_[0], max_[0], 10))

            ax[i].set_xlim([min_[0], max_[0]])
            ax[i].set_ylim([min_[i+1], max_[i+1] + 0.1*max_[i+1]])
            

            ax[i].legend()

    plt.savefig(get_image_name(methods))
    plt.show()


if __name__ == "__main__":

    max_epochs=9

    if len(sys.argv) == 2:
        method = str(sys.argv[-1])
        #print(sys.argv)
    else:
        method = 'hyperband'

    print(f'INFO : Starting method : {method}\n')

    if not os.path.exists(f'results/{max_epochs}'):
        os.makedirs(f'results/{max_epochs}') 

    run_experiment(max_epochs=max_epochs, method=method, test_batch_size=256)
    build_plots(max_epochs=max_epochs)
