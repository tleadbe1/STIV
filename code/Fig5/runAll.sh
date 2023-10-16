#!/usr/bin/bash


echo "running 17 mass simulations"
cd /code/Fig5/Linear17

bash runAllSim.sh
python3 getFrontVel.py
python3 getVisDiss.py

echo "running 62 mass simulations"

cd /code/Fig5/Linear62/V1
python3 runX0.py
python3 runSimSplit.py

echo "plotting"
cd /code/Fig5/
python3 plotAll.py