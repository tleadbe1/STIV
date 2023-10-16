#!/usr/bin/bash

echo "Running simulations and analysis"
i=1
n=10
for f in V1_16 V1_8 V1_4 V1_2 V1 V2 V4 V6 V8 V10
do
  mv computeThermo.py ./$f
  cd ./$f
  python3 runX0.py
  python3 runSimSplit.py
  python3 computeThermo.py
  mv computeThermo.py ../ 
  echo "$i,$n"
  i=$((i+1))
  cd ../
done

