# Project description

### AMS
The starting point of the project are the rare-event algorithms AMS/TAMS in their application to conceptual ocean models. 
These rare-event algorithms rely on the choice of a score function, which quantifies the proximity to the target set for a given trajectory. References can be found below.

### Pullback attractor
Non-autonomous models are dynamical systems with explicit time-dependence. Here, attractors are replaced by Forward and Pullback attractors.
The pullback attractor, in particular, is the area all trajectories starting at $t_0 \to \infty$ converge to. \
As such, it holds promise as a score function for AMS. In the project, we constructed a score function based on the Pullback attractor, implemented it in AMS and applied to the toy model of the time-dependent Double-Well potential.

# Repository structure
The main folder contains the project report as a pdf.

All python scripts can be found in the *scripts* folder:
+ *DoubleWell_model.py* sets up the model, calculates the pullback and implements the integration scheme.
+ It is called by *AMS.py* and *MC.py* which run the AMS-algorithm and straight-up Monte-Carlo simulations, respectively.
+ The available score functions for AMS are specified in *Score_functions.py*. The options are PB-score and simple(~x) score.

Outputs from the scripts are saved as following:
+ Monte-Carlo and AMS results are stored in the *simulations* folder as txt-files. They contain all necessary information, such as model parameters, runtimes and estimated probabilities.
+ Figures are saved to the *plots* folder as png-files.


# Collaborate
````
git clone git@github.com:fabian-lnsm/Pullback-score-AMS.git
````

# References
[1] Jacques-Dumas, V., van Westen, R. M., Bouchet, F., and Dijkstra, H. A.: Data-driven methods to estimate the committor function in conceptual ocean models, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2022-1362, 2022. \
[2] Val´erian Jacques-Dumas, Ren´e M van Westen, and
Henk A Dijkstra. Estimation of amoc transition
probabilities using a machine learning–based rare-
event algorithm. Artificial Intelligence for the Earth
Systems, 3(4):e240002, 2024. \
[3] Peter Kloeden and Meihua Yang. An introduction
to nonautonomous dynamical systems and their at-
tractors, volume 21. World Scientific, 2020 \
[4] Pascal Wang, Daniele Castellana, and Henk Dijk-
stra. Improvements to the use of the trajectory-
adaptive multilevel sampling algorithm for the study
of rare events. Nonlinear Processes in Geophysics
Discussions, 2020:1–24, 2020
