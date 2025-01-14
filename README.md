# Project description
See the [project report](Pullback-Score-AMS.pdf).

# Repository structure
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
