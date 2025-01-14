import numpy as np
import matplotlib.pyplot as plt
import time as time


class DoubleWell_1D:
    """
    Stochastic time-dependent Double-Well potential in 1D.
    """

    def __init__(self, mu : float = 0.03, noise_factor : float = 0.1, dt : float = 0.01, seed=None):

        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.mu = mu
        self.dt = dt
        self.noise_factor = noise_factor
        self.set_roots()

    def reset_seed(self, seed):
        self.rng.bit_generator.state = np.random.PCG64(seed).state    
    
    def is_on(self, traj):
        """
        Checks for on-states in a given set of trajectories.

        """
        time_current = np.round(traj[..., 0], 2)
        root_on = np.vectorize(self.on_dict.get)(time_current)
        on = traj[..., 1] <= root_on #left of the first root (on-state)
        return on

    def is_off(self, traj):
        """
        Checks for off-states in a given set of trajectories.
        """ 
        time_current = np.round(traj[..., 0], 2)
        root_off = np.vectorize(self.off_dict.get)(time_current)
        off = root_off <= traj[..., 1]
        return off
    
    def set_roots(self, all_t = None):
        """
        Set the roots of the system for a given time interval. Necessary to track the on/off states.        
        """
        if all_t is None:
            all_t = np.arange(0, 80, 0.01, dtype=float).round(2) # needs to be tuned if mu changes
        roots = np.real(np.array([np.roots([-1, 0, 1, self.mu*t]) for t in all_t]))
        self.root_times = all_t
        self.on_dict = dict(zip(all_t.T, roots[:, 1])) #left equilibrium point
        self.off_dict = dict(zip(all_t.T, roots[:, 0])) #right equilibrium point

    def number_of_transitioned_traj(self, traj):
        """
        Returns the number of transitions to the off-state for given trajectories.
        """
        sum = np.sum(self.is_off(traj), axis=1)
        transitions = np.sum(sum > 0)
        return transitions

    def potential(self, t, x):
        """
        The potential of the system.
        """
        return x**4 / 4 - x**2 / 2 - self.mu * x * t
    
    def plot_potential(self, time, ax):
        '''
        Plot the potential of the system at a given input time.

        '''
        x = np.linspace(-2, 2, 1000)
        y = self.potential(time, x)
        ax.plot(x, y, label=f't={time}')
        ax.set_xlabel(r' Position x')
        ax.set_ylabel(r'V(x,t)')
        return ax

    def force(self, t, x, mu):
        """
        The force term of the system.
        """
        return -(x**3) + x + mu * t

    def euler_maruyama(
        self, t: np.ndarray, x: np.ndarray, dt: float, mu: float, noise_term: np.ndarray
    ):
        """
        Standard Euler-Maruyama integration scheme for the Double-Well model.
        """
        t_new = t + dt
        drift = self.force(t, x, mu) * dt
        x_new = x + drift + noise_term
        return t_new, x_new

    
    def trajectory_AMS(
            self,
            N_traj: int,
            init_state: np.ndarray,
            downsample: bool = False,
    ) -> tuple [np.ndarray, int]:
        """
        Compute trajectories of the system of variable length (AMS) using Euler-Maruyama method. Returns (traj, transitions).
        """

        traj = []
        traj.append(init_state)
        active_traj = np.arange(N_traj) # Index of the trajectories that are still running
        i = 0
        transitions = 0
        transit_back = 0
        while len(active_traj) > 0:
            noise_term = self.noise_factor * self.rng.normal(loc=0.0, scale=np.sqrt(self.dt), size=len(active_traj))
            t_current, x_current = traj[i][active_traj, 0], traj[i][active_traj, 1]

            t_new, x_new = np.full(N_traj, np.nan), np.full(N_traj, np.nan)
            t_new[active_traj], x_new[active_traj] = self.euler_maruyama(t_current, x_current, self.dt, self.mu, noise_term)
            traj.append(np.stack([t_new, x_new], axis=1))

            back_to_on = ~self.is_on(traj[i][active_traj]) & self.is_on(traj[i+1][active_traj])
            reached_off = self.is_off(traj[i+1][active_traj])
            if np.any(reached_off):
                transitions += np.sum(reached_off)
            if np.any(back_to_on):
               transit_back += np.sum(back_to_on)
            active_traj = active_traj[np.flatnonzero(~(back_to_on | reached_off))] 
            i += 1
        
        traj = np.array(traj)
        traj = np.transpose(traj, (1, 0, 2))

        if downsample==True:
            # Downsample the trajectory to return model time units. Optional.
            i = int(1 / self.dt)
            traj = traj[:, ::i, :]

        return traj, transitions


    def get_pullback(self, return_between_equil: bool = False, N_traj=100, T_max=400, t_0=-200):
        '''
        Calculates Pullback attractor of the system.
        '''

        t_init = np.full(N_traj, t_0)
        x_init = np.linspace(-2, 2, N_traj)
        initial_state = np.stack([t_init, x_init], axis=1)
        traj = self.trajectory_fixedLength(
            N_traj, T_max, initial_state, return_all=True, noise_factor=0
        ) # noise_factor=0 to get deterministic trajectory & return all timesteps
        traj = np.mean(traj, axis=0)
        if return_between_equil==True:
            mask = (traj[:,0] >= 0) & (traj[:,1] <= 1.6)
            traj = traj[mask]
        self.PB_traj = traj
        return traj
    
    def trajectory_fixedLength(
        self,
        N_traj: int,
        T_max: int,
        init_state: np.ndarray,
        return_all = False,
        noise_factor = None,
    ):
        """
        Compute trajectories of the system of fixed length (~TAMS) using Euler-Maruyama scheme.
            
        """
        if noise_factor is None:
            noise_factor = self.noise_factor
        n_steps = int(T_max / self.dt)
        trajectories = np.zeros((N_traj, n_steps, 2))
        t = init_state[:, 0]
        x = init_state[:, 1]
        trajectories[:, 0, 0] = t
        trajectories[:, 0, 1] = x
        noise = noise_factor * self.rng.normal(
            loc=0.0, scale=np.sqrt(self.dt), size=(N_traj, n_steps)
        )
        for i in range(1, n_steps):
            t, x = self.euler_maruyama(t, x, self.dt, self.mu, noise[:, i])
            trajectories[:, i, 0] = t
            trajectories[:, i, 1] = x

        if return_all==False:
            # Downsample the trajectory to return model time units
            i = int(1 / self.dt)
            trajectories = trajectories[:, ::i, :]
        return trajectories
    

    def plot_OnOffStates(self,  ax):
        '''
        Plot the on/off states of the system for a given time interval.
        '''
        on_state = np.vectorize(self.on_dict.get)(self.root_times)
        off_state = np.vectorize(self.off_dict.get)(self.root_times)
        ax.plot(self.root_times, on_state, color='blue')
        ax.plot(self.root_times, off_state, color='darkred')
        y_min, y_max = ax.get_ylim()
        ax.fill_between(self.root_times, on_state, y_min, color='blue', alpha=0.3, label='On-state')
        ax.fill_between(self.root_times, off_state, y_max, color='darkred', alpha=0.3, label='Off-state')
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"Position $x$")
        ax.grid()
        return ax

    def plot_pullback(self, ax):
        '''
        Plot the pullback trajectory of the system.
        '''
        PB_traj = self.get_pullback()
        ax.plot(
                PB_traj[:, 0],
                PB_traj[:, 1],
                label="Pullback attractor",
                color="black", linewidth=2, linestyle='--'
                )
        return ax




if __name__ == "__main__":

    # Plotting potentials at different times: Writes plot to file
    def plot_potentials(times, mu=0.03, initTimes=None, filepath = '../plots/potentials.png'):
        DW_model = DoubleWell_1D(mu=mu)
        fig, ax = plt.subplots(dpi=250)
        ax.set_title(rf'Potential with $\mu = {mu}$')
        for t in times:
            ax = DW_model.plot_potential(t, ax)
        if initTimes is not None:
            init_positions = np.vectorize(DW_model.on_dict.get)(initTimes)
            potentials = DW_model.potential(initTimes, init_positions)
            ax.scatter(init_positions, potentials, color='black', label='Initial States', s=30, zorder=10)
        ax.grid()
        ax.legend()
        fig.savefig(filepath)
        print(f'Figure written to {filepath}')
        plt.close(fig)

    #Plotting phase-space: Writes plot to file
    def plot_phase_space(mu=0.03, plot_OnOff=True, plot_PB=False, initTimes=None, filepath = '../plots/phase_space.png'):
        DW_model = DoubleWell_1D(mu=mu)
        fig, ax = plt.subplots(dpi=250)
        ax.set_title(rf'Phase Space with $\mu = {mu}$')
        if plot_PB==True:
            ax = DW_model.plot_pullback(ax)
        if initTimes is not None:
            init_positions = np.vectorize(DW_model.on_dict.get)(initTimes)
            init_states = np.stack([init_times, init_positions], axis=1)
            ax.scatter(init_states[:,0], init_states[:,1], color='black', label='Initial States', s=30, zorder=10)
        if plot_OnOff==True:
            ax = DW_model.plot_OnOffStates(ax)
        ax.set_xlim(0, 30)
        ax.set_ylim(-1.3, 1.5)
        ax.legend()
        fig.savefig(filepath)
        print(f'Figure written to {filepath}')
        plt.close(fig)

    init_times = np.array([2.0, 4.0, 7.0, 10.0])
    plot_phase_space()
    times = [0, 5, 10, 15, 20]
    plot_potentials(times)

    
