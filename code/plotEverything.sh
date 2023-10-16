#!/usr/bin/bash

echo "Figure 2"
cd /code/Fig2

echo "plotting" 
python3 plotPrev.py
mv Fig2.pdf /results


echo "Figure 3"
cd /code/Fig3
echo "Plotting"
python3 plotDensPrev.py
mv Fig3.pdf /results

echo "Figure 4"
cd /code/Fig4
echo "Plotting"
python3 plotPrev.py
mv Fig4.pdf /results


echo "Figure 5"
cd /code/Fig5
echo "Plotting"
python3 plotAllPrev.py
mv Fig5.pdf /results


cd ../
echo "Finished"