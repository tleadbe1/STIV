#!/usr/bin/bash

echo "Figure 2"
cd /code/Fig2
echo "Running simulations and analysis"
python3 runX0.py
python3 runSims.py
python3 computeMacro.py
echo "plotting" 
python3 plot.py
mv Fig2.pdf /results


echo "Figure 3"
cd /code/Fig3
echo "Running simulations and analysis"
python3 runX0.py
python3 runSims.py  
echo "Plotting"
python3 plotDens.py
mv Fig3.pdf /results

echo "Figure 4"
cd /code/Fig4
bash runAll.sh
mv Fig4.pdf /results

echo "Figure 5"
cd /code/Fig5
bash runAll.sh
mv Fig5.pdf /results


cd ../
echo "Finished"