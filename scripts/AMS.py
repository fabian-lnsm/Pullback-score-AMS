
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

from DoubleWell_Model import DoubleWell_1D
from Score_Functions import score_x, score_PB



class AMS_algorithm():
    
    def __init__(self, noise_factor, mu = 0.03,  dt = 0.01, N_traj = 10000, nc = 100, seed=None):
        self.mu, self.noise_factor = mu, noise_factor
        self.N_traj, self.nc = N_traj, nc
        self.rng = np.random.default_rng(seed=seed)
        self.dimension = 2 # includes time
        self.model = DoubleWell_1D(mu=mu, noise_factor=noise_factor, dt=dt)
        self.traj_function = self.model.trajectory_AMS

    def reset_seed(self, seed):
        self.rng.bit_generator.state = np.random.PCG64(seed).state

    def set_score(self, score_choice : str, PB_decay_length : float = 1.5, clip_onzone=False):
        '''
        Set score function for the AMS algorithm: Choices - 'simple' or 'PB'
        '''
        self.clip_onzone = clip_onzone
        self.score_choice = score_choice
        if score_choice == 'PB':
            self.score_function = score_PB(mu=self.mu, decay_length=PB_decay_length)
        elif score_choice == 'simple':
            self.score_function = score_x()
        else:
            raise ValueError('Score function not recognized. Choose between "simple" and "PB"')
        print(f'Chosen score function: {self.score_function}', flush=True)


    def comp_score(self, traj):
        '''
        Wrapper to call score function.
        '''
        return self.score_function.get_score(traj)

    def comp_traj(self, N_traj : int, init_state : np.array):
        '''
        Wrapper to call trajectory function.
        '''
        traj, _  = self.traj_function(N_traj, init_state)
        return traj
    
    def get_true_length(self, traj):
        '''
        Get true length (without nan) of trajectories arrays
        '''
        return np.where(np.isnan(traj).any(axis=2).any(axis=1), np.argmax(np.isnan(traj).any(axis=2), axis=1), traj.shape[1])

    def run(self, init_state : np.ndarray):
        '''
        Perform one AMS run starting from a given initial state.
            
        '''
        k, w = 0, 1

        traj = self.comp_traj(self.N_traj, init_state)
        max_length = traj.shape[1]
        score = self.comp_score(traj)
        offzone = self.model.is_off(traj)
        score[offzone] = 1
        if self.clip_onzone:
            onzone = self.model.is_on(traj)
            score[onzone] = 0

        Q = np.nanmax(score, axis=1)

        while len(np.unique(Q)) > self.nc:
            threshold = np.unique(Q)[self.nc-1]
            idx, other_idx = np.flatnonzero(Q<=threshold), np.flatnonzero(Q>threshold)
            w *= (1-len(idx)/self.N_traj)
            Q_min = Q[idx]

            # Clone and mutate
            new_ind = self.rng.choice(other_idx, size=len(idx))
            restart = np.nanargmax(score[new_ind]>=Q_min[:,np.newaxis], axis=1)
            init_clone = traj[new_ind,restart,:]
            new_traj = self.comp_traj(len(idx), init_clone)
            max_length_newtraj = np.max(restart + self.get_true_length(new_traj))
            if max_length_newtraj > max_length:
                traj = np.concatenate((traj, np.full((self.N_traj,max_length_newtraj-max_length,self.dimension),np.nan)), axis=1)
                score = np.concatenate((score, np.full((self.N_traj,max_length_newtraj-max_length),np.nan)), axis=1)
                max_length = max_length_newtraj

            # Update trajectories
            for i in range(len(idx)):
                tr_idx, rs, length = idx[i], restart[i], self.get_true_length(new_traj)[i]
                traj[tr_idx,:rs+1,:] = traj[new_ind[i],:rs+1,:]
                traj[tr_idx,rs+1:rs+length,:] = new_traj[i,1:length,:]
                traj[tr_idx,rs+length:,:] = np.nan
                score[tr_idx,:] = self.comp_score(traj[tr_idx,:,:])
                offzone = self.model.is_off(traj[tr_idx])
                score[tr_idx, offzone] = 1
                if self.clip_onzone:
                    onzone = self.model.is_on(traj[tr_idx])
                    score[tr_idx, onzone] = 0

            #Prepare next iteration
            k += 1
            Q = np.nanmax(score,axis=1)

        count_collapse = np.count_nonzero(Q>=1) # transitions = maximum score larger than 1
        print(f'One run succesfull: {count_collapse} transitions', flush=True)
        return dict({
            'probability':w*count_collapse/self.N_traj,
            'iterations':k,
            'nb_transitions':count_collapse,
            })
    
    def plot_trajectories_during_run(self, traj, k):
        '''
        Plot trajectories at AMS iterations (useful to check correctness).
        '''
        fig, ax = plt.subplots(dpi=250)
        for i in range(self.N_traj):
            ax.plot(traj[i,:,0], traj[i,:,1])
        ax.set_title(f'AMS iteration {k}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Position x')
        fig.savefig(f'../temp/traj/AMS_{k}.png')
        plt.close(fig)
    
    def _run_single(self, init_state, seed):
        '''
        Wrapper method for AMS run. Used for parallelelized multiple runs (see run_multiple)
        '''
        self.reset_seed(seed)
        return self.run(init_state)
    
    def run_multiple(self, init_state : np.ndarray, nb_runs : int, filepath : str):
        '''
        Run multiple AMS runs for one given initial state.

        '''
        # preparation
        self.nb_runs = nb_runs
        self.initial_condition = init_state # used for printing and writing
        init_state = np.tile(init_state, (self.N_traj, 1)) # used for simulation

        #plot some information
        print('-'*20, flush=True)
        print('-'*20, flush=True)
        print(f'Initial state: {self.initial_condition}', flush=True)
        print(f'Running {nb_runs} simulations in parallel...', flush=True)

        # Run the simulations
        t_start = time.perf_counter()
        seeds = [np.random.randint(0, 2**16 - 1) for _ in range(self.nb_runs)]
        with Pool() as pool:
            results = pool.starmap(self._run_single, [(init_state, seed) for seed in seeds])
        stats = np.zeros((nb_runs, 3))
        for i, result in enumerate(results):
            stats[i,0] = result['probability']
            stats[i,1] = result['iterations']
            stats[i,2] = result['nb_transitions']
        t_end = time.perf_counter()
        runtime = t_end - t_start

        # print and write to file
        print('-'*20, flush=True)
        print('All simulations finished', flush=True)
        print('Total runtime:', runtime, flush=True)
        print(f'Probability: {np.mean(stats[:,0])} +/- {np.std(stats[:,0])}', flush=True)
        print('-'*20, flush=True)
        print('-'*20, flush=True)
        self.write_results(stats, runtime, filepath)
    
    def write_results(self, stats, runtime, filepath):
        '''
        Writes AMS results to textfile.
        '''
        prob = (np.mean(stats[:,0]), np.std(stats[:,0], ddof=1))
        iter = (np.mean(stats[:,1]), np.std(stats[:,1], ddof=1))
        trans = (np.mean(stats[:,2]), np.std(stats[:,2], ddof=1))
        with open(filepath, 'a') as f:
            f.write(f'Model: g={self.model.noise_factor}, mu={self.model.mu}, runs={self.nb_runs}, N_traj={self.N_traj}, nc={self.nc} \n')
            f.write(f'Score function: {self.score_function} & clip_onzone = {self.clip_onzone} \n')
            f.write(f'Runtime: {runtime}\n')
            f.write(f'Initial state: {self.initial_condition}\n')
            f.write(f'Probability: {prob[0]} +/- {prob[1]}\n')
            f.write(f'Iterations: {iter[0]} +/- {iter[1]}\n')
            f.write(f'Transitions: {trans[0]} +/- {trans[1]}\n')
            f.write('\n')

        
if __name__ == "__main__":
   


    def AMS_simulation(init_times : np.array, score_choice : str, noise_factor : float,
                       nb_runs : float = 20, PB_decay_length : float = 1.5):
        AMS = AMS_algorithm(noise_factor=noise_factor)
        AMS.set_score(score_choice=score_choice, PB_decay_length=PB_decay_length)
        init_positions = np.vectorize(AMS.model.on_dict.get)(init_times)
        init_states = np.stack([init_times, init_positions], axis=1)
        for init_state in init_states:
            AMS.run_multiple(init_state, nb_runs=nb_runs, filepath='../simulations/AMS_results.txt')


    noise = 0.02
    score_function = 'PB'
    init_times = np.array([2.0])
    decay_length = 0.1
    AMS_simulation(init_times=init_times, score_choice=score_function,
                        noise_factor=noise, PB_decay_length=decay_length)

    



   



