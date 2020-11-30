## Optimization for ML - Mini Project

The idea is to implement three hyperparameter optimization algorithms on the MNIST classification set.

We are primarily using three optimization algorithms:

1. Hyperband (State of the art optimization for parameter algorithm)
2. Particle Swarm Optimization (PSO)
3. Population based Training (PBT)

and we are trying to optimizing it over four parameters

1. Activation function (for e.g. relu, selu, tanh)
2. Beta 1 - ADAM Optimizer Parameter
3. Beta 2 - ADAM Optimizer Parameter
4. Learning rate of the ML model

### To run the script with memory profiler:

1. Install `memory_profiler` with `pip install memory_profiler`
2. Run a script with the suitable method: `mprof run experiments.py <name-of-algorithm>` for e.g. `mprof run experiments.py hyperband` (or for PCO). 
3. (TO TRACK CHILD PROCESSES) For PBT, run `mprof run --multiprocess experiments.py pbt` -- so that we can track the memory required by child processes.
4. Once the algorithm is run: move all the files starting with `mprofile_202006*` to a new folder -- so that there are no discrepancies (`mkdir logs_pco && mv mprofile_202006* logs_pco/`).
5. Go to the new folder you made and run the command: `mprof plot --flame`
