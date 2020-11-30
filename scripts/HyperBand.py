import numpy as np
import tensorflow as tf
from utils import build_model, get_data

hp_dictionary = {
    'activation' : ['relu', 'tanh', 'selu'],
    'units_input' : np.linspace(512, 1024, 62),
    'units_hidden' : np.linspace(128, 512, 64),
    'learning_rate' : [0.001, 0.005, 0.01]
}

"""
Hyperband class
max_epoch -- maximal amount of epochs(resource) available for single configuretion
fraction -- controls the proportion of configurations discarded in each round of Successive Halving
"""

class HyperBand:

  def __init__(self, max_epoch = 25, batch_size = 256, model = None, hp_dict = None, fraction = 3.):
    self.R = max_epoch # notation from original paper
    self.eta = float(fraction) # notation from original paper
    self.s_max = int(np.log(self.R)/np.log(self.eta)) # notation from original paper
    self.B = (self.s_max + 1)*self.R # notation from original paper
    self.x_train, self.x_test, self.x_val, self.y_train, self.y_test, self.y_val = get_data()

    self.batch_size = batch_size

    #assert len(hp) == 4, "Print hp len dismatch"
    self.hp_dict = hp_dictionary if hp_dict == None else hp_dict
    self.model = build_model if model == None else model

    self.top_loss_confs = {} # structure {loss_value : (bracket number, conf)}

    """
    for monitoring
    """
    self.monitored_trials = 0 # stores number of times training was run
    self.monitored_epochs = 0 # stores overal number of epochs was run during all trainings
    self.monitored_top_loss = 0
    self.configuration_stats = []

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
    print('In search of Hyperband')
    for s in range(self.s_max + 1)[::-1]:
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
          tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, write_graph=True, profile_batch=2, update_freq='epoch')
          model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=int(r_i), validation_data=(self.x_val, self.y_val), callbacks=[tboard_callback])
          metrics = model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
          
          loss_conf_dict[metrics[0]] = t

          self.monitored_trials += 1 
          self.monitored_epochs += r_i
          self.monitored_top_loss = metrics[0] if metrics[0] > self.monitored_top_loss else self.monitored_top_loss

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
    return self.top_loss_confs[sorted(self.top_loss_confs.keys())[0]][1][0]
  
  def get_monitored_values(self):
    return self.monitored_epochs, self.monitored_trials, self.monitored_top_loss

  def get_all_confs(self):
    return self.configuration_stats
