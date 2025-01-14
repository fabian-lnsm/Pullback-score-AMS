import numpy as np
import time as time
from DoubleWell_Model import DoubleWell_1D


class MonteCarlo:

    def __init__(self, nb_runs : int, noise_factor : float, mu=0.03, seed = None):
        self.nb_runs = nb_runs
        self.noise_factor = noise_factor
        self.mu = mu
        self.model = DoubleWell_1D(mu=mu, noise_factor=noise_factor)
        self.trajectory = self.model.trajectory_AMS
        self.rng = np.random.default_rng(seed=seed)

    def reset_seed(self, seed):
        self.rng.bit_generator.state = np.random.PCG64(seed).state

    def write_results(self, stats, runtime, filepath):
        '''
        Writes MC results to textfile.
        '''
        prob = (np.mean(stats[:,0]), np.std(stats[:,0], ddof=1))
        simulated_traj = (np.mean(stats[:,1]), np.std(stats[:,1], ddof=1))
        transitions = (np.mean(stats[:,2]), np.std(stats[:,2], ddof=1))
        with open(filepath, 'a') as f:
            f.write(f'Model: g={self.noise_factor}, mu={self.mu}, runs={self.nb_runs}\n')
            f.write(f'Runtime: {runtime}\n')
            f.write(f'Initial state: {self.init_state}\n')
            f.write(f'Probability: {prob[0]} +/- {prob[1]}\n')
            f.write(f'Simulated traj: {simulated_traj[0]} +/- {simulated_traj[1]}\n')
            f.write(f'Transitions: {transitions[0]} +/- {transitions[1]}\n')
            f.write('\n')

    def simulate(
        self,
        init_state: np.ndarray,
        N_transitions: int,
        N_traj: int,
        printing: bool = False,
    ):
        '''
        Compute Trajectories until a fixed number of transitions is reached. Returns a dict.
        '''
        if printing==True:
            print('-'*50)
            print(f'Number of transitions: {N_transitions}', flush=True)
            print(f'Number of trajectories per run: {N_traj}', flush=True)

        transitions = 0
        simulated_traj = 0
        time_start = time.perf_counter()
        while transitions < N_transitions:
            _, transit = self.trajectory(N_traj, init_state, downsample=False)
            transitions += transit
            simulated_traj += N_traj
            print(f'Currently: {transitions} transitions. {simulated_traj-transitions} trajectories left', flush=True)
        prob = transitions/simulated_traj
        time_end = time.perf_counter()
        runtime = time_end - time_start

        if printing==True:
            print('-'*50)
            print('Success!')
            print('-'*50)
            print(f'Total runtime: {runtime}')
            print(f'Number of trajectories: {simulated_traj}')
            print(f'Number of transitions: {transitions}')
            print(f'Probability: {prob}')
        
        return dict ({'probability': prob, 'simulated_traj': simulated_traj, 'transitions': transitions})

    def simulate_multiple(self, init_state: np.ndarray, filepath: str, N_transitions: int = 30, N_traj: int = 100000):
        '''
        Runs multiple MC simulations for a given initial state. Writes to file.
        '''

        # prepare initial state
        self.init_state = init_state # used for printing and writing
        init_state = np.tile(init_state, (N_traj, 1)) # used for the simulation

        # print some information
        print('-'*50, flush=True)
        print('-'*50, flush=True)
        print(f'Initial state: {self.init_state}', flush=True)
        print(f'Number of runs: {self.nb_runs}', flush=True)
        print('-'*50, flush=True)

        # Run the simulations consecutively
        t_start = time.perf_counter()
        stats = np.zeros((self.nb_runs, 3))
        seeds = [np.random.randint(0, 2**16 - 1) for _ in range(self.nb_runs)]
        for i, seed in enumerate(seeds):
            print(f'Run {i+1}/{self.nb_runs}', flush=True)
            self.reset_seed(seed)
            result = self.simulate(init_state, N_transitions, N_traj, printing=False)
            print('Run finished!', flush=True)
            stats[i,0] = result['probability']
            stats[i,1] = result['simulated_traj']
            stats[i,2] = result['transitions']            
        t_end = time.perf_counter()
        runtime = t_end - t_start

        # print and write to file
        print('-'*50, flush=True)
        print('All runs finished')
        print('Total runtime:', runtime, flush=True)
        print(f'Probability: {np.mean(stats[:,0])} +/- {np.std(stats[:,0])}', flush=True)
        print('-'*50, flush=True)
        self.write_results(stats, runtime, filepath)


if __name__ == "__main__":


    # Run MC simulations
    def run_MC(init_times, noise_factor, nb_runs = 20):
        MC = MonteCarlo(nb_runs=nb_runs, noise_factor=noise_factor)
        init_positions = np.vectorize(MC.model.on_dict.get)(init_times)
        init_states = np.stack([init_times, init_positions], axis=1)
        for init_state in init_states:
            MC.simulate_multiple(init_state=init_state, filepath='../simulations/MC_results.txt')

    noise = 0.02
    init_times = np.array([2.0, 4.0, 7.0, 10.0])
    run_MC(init_times=init_times, noise_factor=noise, nb_runs=1)



