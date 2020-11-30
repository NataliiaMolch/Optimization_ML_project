from pbt_class import Worker
from multiprocessing import Process, Manager

from random import seed, shuffle
print('Initializing a random seed!')
seed(420)
import itertools

## Get the data function from utils script
import sys
sys.path.append('../')
import utils

import operator
import time
import numpy as np
import matplotlib.pyplot as plt
import logging

"""
Internal Parameters for PBT:

1. Steps (for each step M epoch are run)
2. Population size (for N parallel processes which the process needs to run)
3. Epochs
"""

def run(worker, steps, weight_dict, Q_dict, loss_dict, params_dict, time_dict, trials, epochs, top_loss, monitored_time):
    start_time = time.time()
    print('Worker-{} Inside run function started at: {}'.format(worker.idx, time.time()))
    """start worker object asychronously"""
    for step in range(steps):
        worker.step() # one step of GD
        worker.eval() # evaluate current model
        
        if ((step != 0) and (step in [1, 3])):
            do_explore = worker.exploit()                
            if do_explore:
                print('\n\nInside Explore\n\n')
                worker.explore()
                                        
        worker.update()
    
    ## to let every process instantiate
    time.sleep(worker.idx + 2)
    end_time = time.time()

    ## Update the shared variables with values from each worker
    loss_dict.append({worker.idx: worker.loss_history})
    Q_dict.append({worker.idx: worker.Q_history})
    weight_dict.append({worker.idx: worker.weights_history})
    params_dict.append({worker.idx: [j for j in worker.pop_params[0][worker.idx][1]]})
    time_dict.append({worker.idx: end_time - start_time})

    ## Updating the variables for the integration
    trials.append({worker.idx: worker.monitored_trials})
    epochs.append({worker.idx: worker.monitored_epochs})
    top_loss.append({worker.idx: worker.monitored_top_loss})
    monitored_time.append({worker.idx: worker.time})

class PBT:

    def __init__(self, epochs, parallel_processes, steps):
        self.epochs = int(epochs)
        self.parallel_processes = int(parallel_processes)
        self.steps = int(steps)

        print('Total of {} workers: Each worker runs for: {} steps with each step having {} epochs'.format(
            self.parallel_processes, self.steps, self.epochs))

        self.monitored_trials, self.monitored_epochs, self.monitored_top_loss, self.time = [0.], [0.], [0.], [0.]

        self.top_loss = None
        self.best_conf = None

    def plot(self, title, type, history, steps, population_sizes):
        
        for population_size in population_sizes:
            if type == 'Q':
                plt.plot(history[population_size], lw=0.7, label=str(population_size))
            else:
                plt.scatter(np.arange(0,steps+1), history[population_size], label=str(population_size), s=10)
        
        if type == 'Q':
            plt.axhline(y=1.1, linestyle='dotted', color='k')
            
        axes = plt.gca()
        axes.set_xlim([0,steps])
        if type == 'Q':
            axes.set_ylim([0.0, 1.1])
        
        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel(type)
        plt.legend(loc='upper right')
        
        plt.show()

    def get_monitored_values(self):
        return self.monitored_trials, self.monitored_epochs, self.monitored_top_loss, self.time

    def get_best_conf(self):
        return self.best_conf, self.top_loss

    def search(self, plot=False):
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s.%(msecs)03d %(name)s %(message)s',
            datefmt="%M:%S"
        )

        Q_dict_with_size = {}
        params_dict_with_size = {}
        loss_dict_with_size = {}
        time_dict_with_size = {}

        manager = Manager()

        ## Get training, validation and testing data for training the parallel models
        data_split = utils.get_data()
        print('Training, testing and validation split has been done!')
        
        ## Parameters to be taken are activation function, beta one, beta two, learning rate
        params_comb = [['relu', 'tanh', 'selu'], [0.7, 0.8, 0.9], [0.9, 0.95, 0.999], [0.001, 0.005, 0.1]]
        combinations = list(itertools.product(*params_comb))
        population_sizes = [self.parallel_processes] ## [2, 4, 8, 12, 16, 20, 24]
        
        for population_size in population_sizes:        
            
            pop_score = manager.list() # create a proxy for shared objects between processes
            pop_score.append({})
            
            pop_params = manager.list()
            pop_params.append({})
            
            steps = self.steps
            ## Generating the population with the random combinations
            shuffle(combinations)
            Population = []
            for i in range(population_size):
                combination_chosen = combinations[i]
                worker_init = Worker(
                    idx=i,
                    data_split=data_split,
                    activation=combination_chosen[0],
                    beta_one=combination_chosen[1],
                    beta_two=combination_chosen[2],
                    l_r=combination_chosen[3],
                    epochs=self.epochs,
                    pop_score=pop_score, 
                    pop_params=pop_params,
                    use_logger=False,
                    asynchronous=True, # enable shared memory between spawned processes
                )
                Population.append(worker_init)
            
            ## Initializing the shared variables for information
            weight_dict = manager.list()
            loss_dict = manager.list()
            Q_dict = manager.list()
            params_dict = manager.list()
            time_dict = manager.list()

            ## Initializing another set of variables for information -- this is just for integration
            trials = manager.list()
            epochs = manager.list()
            top_loss = manager.list()
            monitored_time = manager.list()
            
            processes = []
            # create the processes to run asynchronously
            for worker in Population:
                _p = Process(
                    target=run, 
                    args=(
                        worker, steps, weight_dict, Q_dict, loss_dict, params_dict,
                        time_dict, trials, epochs, top_loss, monitored_time))
                processes.append(_p)
            
            ## start the processes and have five second time delay so that all
            ## processes can run at the same time and initalized
            for i in range(population_size):
                processes[i].start()
                time.sleep(5)
            for i in range(population_size): # join to prevent Manager to shutdown
                processes[i].join()
        
            # find agent with best performance
            best_worker_idx = max(pop_score[0].items(), key=operator.itemgetter(1))[0]
            for i, j in pop_score[0].items():
                print('Worker: {} with accuracy: {}'.format(i, j))
            print('\n')
            print('The best worker is: {} out of a population of size: {}'.format(
                best_worker_idx, len(Population)))

            # save best agent/worker for a given population size
            # Q_dict_with_size[population_size] = Q_dict[best_worker_idx][best_worker_idx]
            # loss_dict_with_size[population_size] = loss_dict[best_worker_idx][best_worker_idx]

            ## Update the variables
            self.top_loss = loss_dict[best_worker_idx][best_worker_idx][-1]
            self.best_conf = params_dict[best_worker_idx][best_worker_idx]

            ## Store the parameters at the end of each worker
            # [(i.activation, i.beta_one, i.beta_two, i.l_r) for i in Population]
            # params_dict_with_size[population_size] = params_dict[best_worker_idx][best_worker_idx]
            # time_dict_with_size[population_size] = time_dict[best_worker_idx][best_worker_idx]

            ### Update the variables for integration which have been setup in the constructor
            self.monitored_trials = np.sum([list(i.values())[0] for i in trials], axis=0)
            self.monitored_epochs = np.sum([list(i.values())[0] for i in epochs], axis=0)
            self.monitored_top_loss = top_loss[best_worker_idx][best_worker_idx]
            self.time = np.sum([list(i.values())[0] for i in monitored_time], axis=0) / self.parallel_processes

        # print('\nQ dict with size')
        # # print(Q_dict_with_size)
        # print('\nLoss dict with size')
        # print(loss_dict_with_size)
        # print('\nParams dict with size')
        # print(params_dict_with_size)
        # print('All params used')
        # print(params_dict)
        # print('Time used')
        # print(time_dict_with_size)

        # print('\n\n')
        # print(trials, epochs, top_loss, monitored_time)
        # print(len(self.monitored_trials), len(self.monitored_epochs), len(self.monitored_top_loss), len(self.time))
        # print('\n\n')
        # print('Monitored Trials: {}'.format(self.monitored_trials))
        # print('Monitored Epochs: {}'.format(self.monitored_epochs))
        # print('Monitored Top Loss: {}'.format(self.monitored_top_loss))
        # print('Time: {}'.format(self.time))

        ## Plot the graph
        if plot:
            plot('Q (Accuracy) per step for various population sizes', 'Q', Q_dict_with_size, steps, population_sizes)
            plot('loss per step for various population sizes', 'loss', loss_dict_with_size, steps, population_sizes)
    
# if __name__ == '__main__':
#     search(plot=True)