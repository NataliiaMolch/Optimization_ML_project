import numpy as np
import operator
import matplotlib.pyplot as plt
import logging

import time
import random
import itertools
import tensorflow as tf


class Worker(object):
    def __init__(self, idx, data_split, activation, beta_one, beta_two, 
        l_r, epochs, pop_score, pop_params, use_logger, asynchronous=False):

        self.idx = idx
        ## Get the data splits and split them into their suitable partitions
        self.x_train, self.x_test, self.y_train, self.y_test = data_split
        
        self.use_logger = use_logger
        if use_logger:
            self.logger = logging.getLogger("Worker-{}".format(self.idx))
        else:
            print("Beginning Worker-{}".format(self.idx))
        
        ## Build model based on the parameters
        self.activation = activation
        self.beta_one = beta_one
        self.beta_two = beta_two
        self.l_r = l_r
        self.model = None
        self.asynchronous = asynchronous
        ## Get weights about the model, initialized to none initially
        self.weights = None
        self.score = 0
        self.loss = 0
        self.epochs = epochs
        
        ## reference to population statistics
        self.pop_score = pop_score
        self.pop_params = pop_params
        
        ## for plotting and storing information
        self.weights_history = []
        self.Q_history = []
        self.loss_history = []

        self.start_time = time.time()
        self.top_loss = float('inf')
        self.monitored_trials, self.monitored_epochs, self.monitored_top_loss, self.time = [0.], [0.], [0.], [0.]
        
        self.update() # intialize population

    def monitor(self):
        self.time.append(time.time() - self.start_time)
        self.monitored_trials.append(self.monitored_trials[-1] + 1)
        self.monitored_epochs.append(self.monitored_epochs[-1] + self.epochs)
        self.monitored_top_loss.append(self.top_loss)

    def step(self):

        def build_model():
            """
            params: hp -- list of all hyperparameters' values, by default:
              hp0 -- 'activation' with possible values ['relu', 'tanh', 'selu']
              hp1 -- 'units_input' with possible values np.linspace(512, 1024, 62)
              hp2 -- 'units_hidden' with possible values np.linspace(128, 512, 64)
              hp3 -- 'learning_rate' with possible values [0.001, 0.005, 0.01]

            Returns the instantiated object of the model.
            """
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.models import Sequential
            import tensorflow as tf
            tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())

            model = Sequential()
            model.add(Dense(512, input_dim=784, activation=self.activation))
            model.add(Dense(256, activation=self.activation))

            ## Using the softmax distribution at the end since we need evaluation
            model.add(Dense(10, activation='softmax'))
            model.compile(
                optimizer=Adam(lr=self.l_r, beta_1=self.beta_one, beta_2=self.beta_two), 
                loss='categorical_crossentropy', metrics=['categorical_accuracy'])
            return model

        self.model = build_model() if self.model is None else self.model
        ## Batch size is 256 in coordination with PCO and Hyperband
        # self.tboard_callback = tf.keras.callbacks.TensorBoard(
        #     log_dir=f'logs_pbt/{self.start_time}_{self.activation}_{self.beta_one}_{self.beta_two}_{self.l_r}',
        #     histogram_freq=1, profile_batch=2)
        history = self.model.fit(
            self.x_train, self.y_train, epochs=self.epochs, 
            batch_size=256, shuffle=True, verbose=0)##, callbacks=[self.tboard_callback])
        self.loss = history.history['loss'][0]
        self.weights = self.model.get_weights()
        print('Weight info: for worker: {} with shape: new: {} old:{}'.format(
            self.idx, len(self.model.get_weights()), len(self.weights)))

    def eval(self):
        """metric we want to optimize -- validation set performance"""
        model_loss, self.score = self.model.evaluate(self.x_test, self.y_test, batch_size=256, verbose=1)
        if model_loss < self.top_loss:
            self.top_loss = model_loss
        self.monitor()
        return self.score
        
    def exploit(self):
        """copy weights, hyperparams from the member in the population with the highest performance"""
        if self.asynchronous:
            pop_score, pop_params = self.proxy_sync(pull=True)
        else:
            pop_score = self.pop_score
            pop_params = self.pop_params
            
        best_worker_idx = max(pop_score.items(), key=operator.itemgetter(1))[0]
        print('\n\n Best worker idx: {} and self index: {}'.format(best_worker_idx, self.idx))
        if best_worker_idx != self.idx:
            # print(self.idx, pop_score) enable to check if shared memory is being updated
            
            ## Setting the model new weights which have performed the best
            best_worker_weights, best_worker_h = pop_params[best_worker_idx]
            ## Two ways to change the weights of the models -- based on if we consider
            ## biases or not in the model change
            self.model.set_weights(best_worker_weights)
            # self.model.layers[0].set_weights(best_worker_weights[0])
            # self.model.layers[1].set_weights(best_worker_weights[2])
            # self.model.layers[2].set_weights(best_worker_weights[4])
            
            if self.use_logger:
                self.logger.info("Inherited optimal weights from Worker-{}".format(best_worker_idx))
            else:
                print("Worker-{} Inherited optimal weights from Worker-{}".format(self.idx, best_worker_idx))
            return True
        return False

    def get_activation_function(self, activation_function):
        from tensorflow.keras import activations

        if activation_function == 'selu':
            return activations.selu
        elif activation_function == 'relu':
            return activations.relu
        else:
            return activations.tanh
        
    def explore(self):
        """perturb hyperparameters with a random choice again"""
        params_comb = [['relu', 'tanh', 'selu'], [0.7, 0.8, 0.9], [0.9, 0.95, 0.999], [0.001, 0.005, 0.1]]
        combinations = list(itertools.product(*params_comb))
        newly_chosen = random.choice(combinations)
        print('Inside explore: Old activation - {}, Adam beta one - {}, Adam beta two - {} and learning rate - {}'.format(
            self.activation, self.beta_one, self.beta_two, self.l_r))
        newly_chosen_function = self.get_activation_function(newly_chosen[0])

        ## change the number of trials we have -- for the final elements of the list
        ## only update when the newly chosen one is different from the previous one
        if (
            (newly_chosen_function != self.activation) or (newly_chosen[1] != self.beta_one)
            or (newly_chosen[2] != self.beta_two) or (newly_chosen[3] != self.lr)):
            self.monitored_trials[-1] = self.monitored_trials[-1] + 1
        self.beta_one = newly_chosen[1]
        self.beta_two = newly_chosen[2]
        self.l_r = newly_chosen[3]

        ## Assigning the values
        self.activation = newly_chosen_function
        self.model.layers[0].activation = self.activation
        self.model.layers[1].activation = self.activation
        self.model.optimizer.lr.assign(self.l_r)
        self.model.optimizer.beta_1.assign(self.beta_one)
        self.model.optimizer.beta_2.assign(self.beta_two)
        print('Finshing with explore: New activation - {}, Adam beta one - {}, Adam beta two - {} and learning rate - {}'.format(
            self.activation, self.beta_one, self.beta_two, self.l_r))
        
    def update(self):
        """update worker stats in global dictionary"""
        if not self.asynchronous:
            self.pop_score[self.idx] = self.score
            self.pop_params[self.idx] = (
                np.copy(self.weights), np.array([
                    self.activation, self.beta_one, self.beta_two, self.l_r
                ])
            ) 
        else:
            self.proxy_sync(push=True)
            
        self.weights_history.append(np.copy(self.weights))
        self.Q_history.append(self.score)
        self.loss_history.append(self.loss)
        
        if len(self.Q_history) % 10 == 0:
            if self.use_logger:
                self.logger.info("Q = {:0.2f} ({:0.2f}%)".format(self.score, self.score * 100))
            else:
                print("Worker-{} Step {} Q = {:0.2f} ({:0.2f}%)".format(
                    self.idx, len(self.Q_history), self.score, self.score * 100))
                                                                                      
    def proxy_sync(self, pull=False, push=False):
        """
        for asynchronous workers, we need to sync the values to the shared proxies
        """
        if pull: # grab newest copy of pop_params
            return self.pop_score[0], self.pop_params[0]

        if push: # update newest copy
            _pop_score = self.pop_score[0]
            _pop_params = self.pop_params[0]
            
            _pop_score[self.idx] = self.score
            _pop_params[self.idx] = (
                np.copy(self.weights), np.array([
                    self.activation, self.beta_one, self.beta_two, self.l_r
                ])
            ) 
            self.pop_score[0] = _pop_score
            self.pop_params[0] = _pop_params  
