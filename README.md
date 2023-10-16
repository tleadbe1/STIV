# STIV
(Stochastic thermodynamics with internal variables)

Supporting code for manuscript submission to PNAS Nexus named: A stastical mechanics framework for constructing non-equilibrium thermodynamic models.

/environment/ 
The docker file contained in this directory gives the python and ubuntu dependencies needed to run this code. 

/metadata/ 
Contains author information

/code/ 
Contains run and runEverything.sh which will enter into the subdirectories and initiate all simulations. Each subdirectory contains the code to produce the associated figure 
(eg. Fig2 produces Figure 2 in the manuscript). DyanmicsSolvers contains python class files to run and store data associated with the Langevin and STIV simulations. 
