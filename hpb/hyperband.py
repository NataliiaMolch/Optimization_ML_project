import sys
sys.path.append('../')
import utils

import time
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations

class HyperBand:

  def __init__(self, max_epoch = 25, test_batch_size = 256):
    self.R = max_epoch # notation from original paper
    self.eta = 3. # notation from original paper
    self.s_max = int(np.log(self.R)/np.log(self.eta)) # notation from original paper
    self.B = (self.s_max + 1)*self.R # notation from original paper
    self.x_train, self.x_test, self.y_train, self.y_test = utils.get_data()

    self.batch_size = test_batch_size

    #assert len(hp) == 4, "Print hp len dismatch"
    self.hp_dict = utils.hp_dictionary
    self.model = utils.model

    self.top_loss_confs = {} # structure {loss_value : (bracket number, conf)}

    self.top_loss = float('inf')

    """
    for monitoring
    
    self.monitored_trials = 0 # stores number of times training was run
    self.monitored_epochs = 0 # stores overal number of epochs was run during all trainings
    self.monitored_top_loss = 0
    self.configuration_stats = []
    """
    self.monitored_trials, self.monitored_epochs, self.monitored_top_loss, self.time = [0.], [0.], [0.], [0.]  # stores number of times training was run
    self.start_time = time.time()


  def monitor(self, epochs):
    self.time.append(time.time() - self.start_time)
    self.monitored_trials.append(self.monitored_trials[-1] + 1)
    self.monitored_epochs.append(self.monitored_epochs[-1] + epochs)
    self.monitored_top_loss.append(self.top_loss)

  """
  returns list on n hyperparameter configurations.
  each element of the list contains list of hp values in the right order.
  we use uniform distribution.
  """
  def get_hyperparameter_confs(self, n):
    confs = []
    for hp_name in list(self.hp_dict.keys()):
      confs.append(np.random.choice(self.hp_dict[hp_name], n))
    return list(map(list, zip(*confs)))

  """
  returns k confs with smallest losses -- used in 'search' method only
  """
  def top_k_from_dict(self,l_c_dict, k):
    top_k_losses = sorted(list(l_c_dict.keys()))[:k]
    return [l_c_dict[l] for l in top_k_losses], top_k_losses
  
  """
  Called from search() to add best found configurations for this bracket
  """
  def update_best_conf(self, T, s, top_k_losses):
     for bl in top_k_losses:
       self.top_loss_confs[bl] = (s, T)

  def search(self):

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs_hpb/{self.start_time}', histogram_freq=1, profile_batch=2)

    for s in range(self.s_max+1)[::-1]:
      n = int(self.B * pow(self.eta, s)/self.R/(s+1)) + 1
      r = self.R * pow(self.eta, -s)
      T = self.get_hyperparameter_confs(n)
      assert len(T) == n
      """
      SuccessiveHalving(n,r)
      """
      print('Started working with bracket # ',s)
      for i in range(s+1): 
        n_i = int(n*pow(self.eta, -i))
        r_i = r * pow(self.eta, i) # number of epochs (resources) for current configurations
        assert r_i >= 1, "r_i is less than 1"
        loss_conf_dict = {} # has strusture {loss_value : [conf]}
        for t in T: # for each generated configuration evalate performance
          model = self.model(t)

          logs = 'logs/{}_{}_{}'.format(t[0], t[1], t[2])
          ## histogram_freq at which to compute activation and weight histograms
          
          model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=int(r_i), callbacks=[tboard_callback])
          metrics = model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
          
          loss_conf_dict[metrics[0]] = t

          self.top_loss = metrics[0] if metrics[0] < self.top_loss else self.top_loss
          self.monitor(r_i)

        T, top_k_losses = self.top_k_from_dict(loss_conf_dict, int(n_i / self.eta))
      
      self.update_best_conf(T, s, top_k_losses)

    return True
      

  """
  Displays parameters of best configurations and corresponding losses.
  n -- number of top configurations to show. 
  """
  def show_best_configurations(self, n = None):
    itr = 1
    for bl in sorted(self.top_loss_confs.keys()):
      print('Top {num} model, from {s} bracket reached loss {l} has hyperparameters:'.format(num = itr, s = self.top_loss_confs[bl][0], l = bl))
      i = 0
      for hp in self.hp_dict.keys():
        conf = self.top_loss_confs[bl][1]
        print('\t{} = {}'.format(hp, conf[0][i]))
        i+=1
      itr += 1

      if n!= None and itr == n + 1:
        break
    
  """
  Returns best conf
  """
  def get_best_conf(self):
    return self.top_loss_confs[sorted(list(self.top_loss_confs.keys()))[0]][1][0], sorted(list(self.top_loss_confs.keys()))[0]
  
  def get_monitored_values(self):
    return self.monitored_trials, self.monitored_epochs, self.monitored_top_loss, self.time


