import numpy as np
import gymnasium as gym

params_shape = [8, 16, 4]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def reconstruct_weights(layer_sizes, parameters):
    weights = []
    biases = []
    p = 0

    for i in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[i], layer_sizes[i+1]

        size_w = n_in * n_out
        w = parameters[p : p + size_w].reshape(n_in, n_out)
        weights.append(w)
        p += size_w

        size_b = n_out
        b = parameters[p : p + size_b] 
        biases.append(b)
        p += size_b
    
    return (weights, biases)

def calc_number_parameters():
    layer_sizes = params_shape
    t = 0
    for i in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[i], layer_sizes[i+1]
        t += (n_in * n_out) + n_out

    return t

class IndividualNN:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, x):
        activation = x
        # Recorremos todas las capas menos la última con ReLU
        for i in range(len(self.weights) - 1):
            activation = relu(np.dot(activation, self.weights[i]) + self.biases[i])
        
        # Última capa con Sigmoid
        final_z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        return sigmoid(final_z)
    
def init_worker():
    global _worker_env
    _worker_env = gym.make("LunarLander-v3")

def evaluate_solution(content):
    seeds, solution = content

    global _worker_env
    
    w, b = reconstruct_weights(params_shape, solution)

    ind = IndividualNN(w, b)

    solutions_tests = []
    
    for seed in seeds:
        observation, _ = _worker_env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0
        state = observation

        while not (terminated or truncated):
            ind_res = ind.forward(state)
            action = np.argmax(ind_res)
            
            next_observation, reward, terminated, truncated, info = _worker_env.step(action)
            state = next_observation
            
            total_reward += reward

        solutions_tests.append(total_reward)
    
    return -1*np.mean(solutions_tests)
    
def evaluate_individuals(pool, solutions):
    seeds = [np.random.randint(0, 5000) for _ in range(5)]
    scores = pool.map(evaluate_solution, [(seeds, solution) for solution in solutions])
    return scores