# STIV
(Stochastic thermodynamics with internal variables)

Supporting code for the paper: Leadbetter, Travis, Prashant K. Purohit, and Celia Reina. "A statistical mechanics framework for constructing nonequilibrium thermodynamic models." PNAS Nexus 2.12 (2023): pgad417.

/environment/ 
The docker file contained in this directory gives the python and ubuntu dependencies needed to run this code. 

/metadata/ 
Contains author information

/code/ 
Contains run and runEverything.sh which will enter into the subdirectories and initiate all simulations. Each subdirectory contains the code to produce the associated figure 
(eg. Fig2 produces Figure 2 in the manuscript). DyanmicsSolvers contains python class files to run and store data associated with the Langevin and STIV simulations. 
