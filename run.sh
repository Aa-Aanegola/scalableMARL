#!/bin/bash
#SBATCH --job-name=scaleMARL
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=5
#SBATCH --output output.txt
#SBATCH -G 1 -c 10
#SBATCH --mail-type=ALL --mail-user=aakash.aanegola@students.iiit.ac.in

#module load cuda/11.0
#module load cudnn/7-cuda-11.0


source setup
python3 algos/maTT/run_script.py --seed 0 --log_dir ./results/maql-newrew --epochs 40
