#!/bin/bash
#SBATCH --job-name=scaleMARL
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=10
#SBATCH --output output.txt
#SBATCH -G 1 -c 10
#SBATCH --mail-type=ALL --mail-user=aakash.aanegola@students.iiit.ac.in
#SBATCH --mem=30G
#module load cuda/11.0
#module load cudnn/7-cuda-11.0


source setup
python3 algos/maTT/run_script.py --seed 0 --log_dir ./results/dql-newrew --epochs 40
python3 algos/maTT/run_script.py --mode test --log_dir ./results/dql/main/seed_0/ --nb_test_eps 50
python3 algos/maTT/run_script.py --mode test --log_dir ./results/dql-newrew/main/seed_0/ --nb_test_eps 50

# python3 algos/maTT/run_script.py --mode test --render 1 --log_dir ./results/maTT/setTracking-v0_123456789/seed_0/ --nb_test_eps 50
