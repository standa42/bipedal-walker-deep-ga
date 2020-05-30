#!/bin/bash
# this script can be run on cluster using command e.g.: qsub -l mem_free=8G,act_mem_free=8G,h_vmem=12G -cwd -j y sh run.sh
/usr/bin/python3 train.py --nn_width=50 --population_size=120 --seed=106 --generations_count=100000 --max_episode_length=1000