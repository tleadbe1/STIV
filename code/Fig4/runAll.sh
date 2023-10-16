#!/usr/bin/bash
echo "Running simulations and analysis"

for f in Linear SineSlow OffSet Step
do
  echo $f
  cd ./$f
  python3 runX0.py
  python3 runSims.py
  python3 computeThermo.py
  cd ../
done

echo "Plotting results"
python3 plot.py
