import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from DoubleWell_Model import DoubleWell_1D


class score_x:
    """
    Simple score function ( ~(x+1)/2 ).
    """

    def __init__(self):
        self.equilibrium = 1  # abs(x-coordinate) of the equilibrium states

    def __str__(self):
        return f'Simple x-score'
    
    def __repr__(self):
        return f'Simple x-score'

    def get_score(self, traj : np.array):
        """
        Returns score for given set of trajectories.
        """
        x_value = traj[..., 1]
        score = (x_value + self.equilibrium) / (2 * self.equilibrium)
        score = np.clip(score, a_min=None, a_max=1)
        return score

class ScoreFunction_helper:
    '''
    Helper class for the score function. Needed for multiprocessing
    '''
    def __init__(self, reference_traj, normalised_curvilinear_coordinate, decay_length):
        self.kdtree = KDTree(reference_traj)
        self.normalised_curvilinear_coordinate = normalised_curvilinear_coordinate
        self.decay_length = decay_length

    def __call__(self, traj: np.array):
        original_shape = traj.shape[:-1]
        traj_reshape = traj.reshape(-1, 2)
        scores = np.full(traj_reshape.shape[0], np.nan)
        valid_mask = ~np.isnan(traj_reshape).any(axis=1)
        if valid_mask.any():
            closest_distances, closest_point_indices = self.kdtree.query(traj_reshape[valid_mask])
            s = self.normalised_curvilinear_coordinate[closest_point_indices]
            scores[valid_mask] = s * np.exp(-(closest_distances / self.decay_length) ** 2)
        scores = np.clip(scores, a_min=None, a_max=1)
        scores = scores.reshape(original_shape)
        return scores

class score_PB:
    '''
    Pullback attractor as a score function
    '''
    def __init__(self, decay_length, mu=0.03):
        self.decay_length = decay_length
        self.model = DoubleWell_1D(mu=mu)
        self.PB_trajectory = self.model.get_pullback(return_between_equil=True)
        self.score_function = self.score_function_maker(self.PB_trajectory, self.decay_length)

    def __str__(self):
        return f'PB-score (DL={self.decay_length})'
    
    def __repr__(self):
        return f'PB-score (DL={self.decay_length})'

    def curvilinear_coordinate(self, reference_traj):
        '''
        Find the curvilinear coordinate of a trajectory.
        '''
        nb_points = np.shape(reference_traj)[0]
        ds_2 = np.zeros(nb_points - 1)
        for i in range(1, reference_traj.ndim):
            dxi = reference_traj[1:, i] - reference_traj[:-1, i]
            ds_2 = ds_2 + dxi ** 2

        curvilinear_coordinate = np.zeros(nb_points)
        curvilinear_coordinate[1:] = np.cumsum(np.sqrt(ds_2))
        normalised_curvilinear_coordinate = curvilinear_coordinate / curvilinear_coordinate[-1]

        return normalised_curvilinear_coordinate

    def score_function_maker(self, reference_traj, decay_length):
        '''
        Create a score function based on the pullback attractor.

        '''
        normalised_curvilinear_coordinate = self.curvilinear_coordinate(reference_traj)
        return ScoreFunction_helper(reference_traj, normalised_curvilinear_coordinate, decay_length)

    def get_score(self, traj):
        return self.score_function(traj)

    
        
if __name__=='__main__':

    

    def plot_PB_score(decayLength, plot_PB=True, filepath='../plots/PB_score'):
        fig, ax = plt.subplots(dpi=250)
        scorefct_PB = score_PB(decay_length = decayLength)
        t = np.linspace(0, 60, 300) #range for which to calculate the score
        x = np.linspace(-1.7, 1.7, 300) #range for which to calculate the score
        T, X = np.meshgrid(t, x)
        phase_space = np.stack((T, X), axis=-1)
        scores = scorefct_PB.get_score(phase_space)
        scores = np.where(scores == 0, 1e-22, scores)
        ax.set_title(rf'Pullback score with $\lambda = {scorefct_PB.decay_length:.2f}$')
        contour=ax.contourf(
            T, X, scores, cmap='viridis', levels=np.linspace(0,1,51), alpha=0.85
            )
        cbar=fig.colorbar(contour)
        cbar.ax.set_ylabel(r'$\phi_{PB}$', loc='center', labelpad=12)
        if plot_PB==True:
            ax = scorefct_PB.model.plot_pullback(ax)
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"Position $x$")
        ax.set_xlim(0, 25)
        ax.set_ylim(-1.3, 1.5)
        _, labels = ax.get_legend_handles_labels()
        if labels != []:
            ax.legend()
        filepath = filepath + f'({decayLength})'+'.png'
        fig.savefig(filepath)
        print(f'Figure saved to {filepath}')
        plt.close(fig)

    decay_length = 2.5
    plot_PB_score(decay_length, plot_PB=True)

        


