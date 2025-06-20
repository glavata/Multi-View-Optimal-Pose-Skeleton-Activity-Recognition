# Multi-View Optimal Pose Skeleton Activity Recognition
Implementation of the paper "Human Activity Recognition for Real-Time Multi-View Applications with Skeleton Data".
Paper: PDF

## Dependencies
- Python >= 3.11.7
- scikit-learn >= 1.4.0
- scipy >= 1.12.0
- tensorflow >= 2.15.0
- numpy >= 1.26.3
- pomegranate >= 0.14.9
- pandas >= 2.2.0
- tables >= 3.9.2
- matplotlib >= 3.8.2
- dtaidistance >= 2.3.11


## Directory Structure
- seq_gen.py - generator for skeleton activity data; supports NTU-RGB, PKU-MMD; translates and rotates data;
- dataset_gen.py - additional secondary generator for multi-view data merging
- hmm.py and st_gcn.py - method files for Hidden Markov Model and ST-GCN (not finished)
- kalman.py / multikalman - Kalman Filter library which includes a python and C (ctypes) implementation; will be separated from this repository in the future;
- visualizer.py - code for matplotlib visualization of skeletons and activities related
- multi_view_util.py - frame by frame skeleton rotation and merging
- main.py - starts different tasks (classification, segmentation) after setting up generators for the datasets

## Downloading Data


## Running tasks
