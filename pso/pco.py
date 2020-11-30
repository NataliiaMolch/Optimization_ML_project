import sys
sys.path.append('../')
import utils

import time
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations


W = 0.5
c1 = 0.8
c2 = 0.9

activation = {
    1. : 'relu', 2. : 'selu', 3. : 'tanh'
}

class Particle():
    def __init__(self):
        self.hp_dict = utils.hp_dictionary
        self.hp_dict['activation'] = [1., 2., 3.]

        self.position = np.array([random.sample(list(self.hp_dict[name]),1)[0] for name in sorted(list(self.hp_dict.keys()))])
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity =  np.zeros(len(self.position), dtype= 'float')

        self.low, self.high = [0.,0.,0.,0.],[0.,0.,0.,0.]
        self.overal = 0.

        self.model = utils.model()
        self.update_model()

    def move(self):
        self.velocity[0] = int(self.velocity[0])

        self.overal +=1
        self.position += self.velocity


        for i, hp_name in enumerate(sorted(list(self.hp_dict.keys()))):

          pos = self.position[i]

          if pos> max(self.hp_dict[hp_name]):
            self.position[i] = max(self.hp_dict[hp_name])
            self.high[i] += 1
          elif pos < min(self.hp_dict[hp_name]):
              self.position[i] = min(self.hp_dict[hp_name])
              self.low[i] += 1
          else:

            for j, val in enumerate(sorted(self.hp_dict[hp_name])[:-1]):
              next_val = sorted(self.hp_dict[hp_name])[j+1]
              if pos > val and pos < next_val:
                self.position[i] = val if pos - val < next_val - pos else next_val


        self.update_model()

    def get_activation_function(self, activation_function):
	    """ Get activation function signature based on the string"""

	    if activation_function == 'selu':
	        return activations.selu
	    elif activation_function == 'relu':
	        return activations.relu
	    else:
	        return activations.tanh

    def update_model(self):
        ## FIXED: There was a different character like a unicode character
        self.model.optimizer.lr.assign(self.position[3])
        self.model.optimizer.beta_1.assign(self.position[1])
        self.model.optimizer.beta_2.assign(self.position[2])
        self.model.layers[0].activation = self.get_activation_function(activation[int(self.position[0])])
        self.model.layers[1].activation = self.get_activation_function(activation[int(self.position[0])])


"""
PCO:

args = [n_iterations, n_epochs, n_particles]

"""

class PCO:

  def __init__(self, args, test_batch_size = 256):
    self.n_iterations = args[0]
    self.epochs = int(args[1])
    self.n_particles = args[2]

    self.hp_dict = utils.hp_dictionary
    self.hp_dict['activation'] = [1., 2., 3.]

    self.particles = [Particle() for _ in range(self.n_particles)]
    self.gbest_value = float('inf')
    self.gbest_position = [random.sample(list(self.hp_dict[name]),1)[0] for name in list(self.hp_dict.keys())]

    global W, c1, c2
    self.W = W
    self.c1 = c1
    self.c2 = c2
    
    self.x_train, self.x_test, self.y_train, self.y_test = utils.get_data()

    self.batch_size = test_batch_size

    #assert len(hp) == 4, "Print hp len dismatch"
    

    """
    for monitoring
    """
    self.monitored_trials, self.monitored_epochs, self.monitored_top_loss, self.time = [0.], [0.], [0.], [0.]  # stores number of times training was run
    self.start_time = time.time()

  
  def print_outliers(self):
    low = [0.,0.,0., 0.]
    high = [0.,0.,0.,0.]
    overal = 0
    for p in self.particles: ## FIXED: from partictes - self.particles
      low = [low[k] + p.low[k] for k in range(4)]
      high = [high[k] + p.high[k] for k in range(4)]
      overal += p.overal
    print('HP\tLOW\tHIGH\nHP_1\t{}\t{}\nHP_2\t{}\t{}\nHP_3\t{}\t{}\nHP_4\t{}\t{}\n'.format(low[0], high[0], low[1], high[1], low[2], high[2], low[3], high[3]))

  def monitor(self):
    self.time.append(time.time() - self.start_time)
    self.monitored_trials.append(self.monitored_trials[-1] + 1)
    self.monitored_epochs.append(self.monitored_epochs[-1] + self.epochs)
    self.monitored_top_loss.append(self.gbest_value)


  def search(self):
    k = 0
    # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs_pco/{self.start_time}', histogram_freq=1, profile_batch=2)
    while k < self.n_iterations:
        for p in self.particles:
            print('\n\nact:{} b1:{} b2:{} LR:{}'.format(p.position[0], p.position[1], p.position[2], p.position[3]))
            p.model.fit(self.x_train, self.y_train, batch_size = self.batch_size, epochs = self.epochs)#, callbacks=[tboard_callback])

            metrics = p.model.evaluate(self.x_test, self.y_test, batch_size = self.batch_size)
            if metrics[0] < self.gbest_value:
              self.gbest_value = metrics[0]
              self.gbest_position = p.position
            if metrics[0] < p.pbest_value:
              p.pbest_value = metrics[0]
              p.pbest_position = p.position
            
            self.monitor()

        for p in self.particles:
            p.velocity = self.W*p.velocity + (self.c1*random.random()) * (p.pbest_position - p.position) + \
                            (random.random()*self.c2) * (self.gbest_position - p.position)

            p.move()
          
        k = k + 1
    #self.print_outliers()
    return True
    
  """
  Returns best conf
  """
  def get_best_conf(self):
    return self.gbest_position, self.gbest_value
  
  def get_monitored_values(self):
    return self.monitored_trials, self.monitored_epochs, self.monitored_top_loss, self.time

