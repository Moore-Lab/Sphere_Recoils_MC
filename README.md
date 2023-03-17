# Sphere_Recoils_MC

Code to simulate the containment of daughter nuclear recoils following alpha decays in a silica sphere.

The basic simulation steps are as follows:
 1) Generate a set of decay chains to simulate using SRIM 2013 (these are the data files in the decay_data folder)
 2) Use run_SRIM_batch.ipynb Jupyter notebook to generate input files for SRIM
 3) Setup SRIM 2013 to run in batch mode. Then run the cell in run_SRIM_batch.ipynb to loop over input files
 4) Use the simulate_recoil_range.ipynb Jupyter notebook to collect these files, and then simulate the sphere trajectories for a given decay chain. This script also contains some useful plotting functions.
 
 Example simulated decay chain for Ac-225 in a SiO2 sphere:
 ![Alt text](plots/examp_traj_Ac-225_SiO2_5.png?raw=true "Example simulation")
