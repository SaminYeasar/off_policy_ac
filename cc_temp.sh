#!/usr/bin/env bash


envname=("HalfCheetah-v1")
#envname=("Hopper-v2" "Walker2d-v2")
# envname=("Swimmer-v1" "Reacher-v1" "InvertedDoublePendulum" "InvertedPendulum")

#methods=("DDPG" "TD3" "SAC")

methods=("DDPG" "TD3")
#seeds=(0 1 2 3 4 5 6 7 8 9)

lmbdas=("0.0" "0.2" "0.5" "0.8")

seeds=(0)

for method in ${methods[@]}
do
  for env in ${envname[@]}
  do
    for seed in ${seeds[@]}
    do
        for lmbda in ${lmbdas[@]}
        do
          echo "#!/bin/bash" >> temprun.sh
          echo "#SBATCH --account=def-dprecup" >> temprun.sh
          echo "#SBATCH --output=\"/scratch/syarnob/off_ac/slurm-%j.out\"" >> temprun.sh
          echo "#SBATCH --job-name=off_ac_env_${env}_method_${method}" >> temprun.sh
          echo "#SBATCH --cpus-per-task=6"  >> temprun.sh
          echo "#SBATCH --gres=gpu:1" >> temprun.sh
          echo "#SBATCH --mem=10G" >> temprun.sh
          echo "#SBATCH --time=4:00:00" >> temprun.sh
          echo "module load singularity" >> temprun.sh
          echo "echo "Syncing singularity image..." " >> temprun.sh
          echo "cp -r /scratch/syarnob/rllab.simg \$SLURM_TMPDIR/" >> temprun.sh
          echo "echo "Running image." " >> temprun.sh
          echo "singularity exec --nv -B \$SLURM_TMPDIR:/tmp \$SLURM_TMPDIR/rllab.simg python main.py --policy_name ${method[@]} --env_name ${env[@]} --seed ${seed[@]} --lmbda ${lmbda[@]}" >> temprun.sh
                  cat temprun.sh
                  eval "sbatch temprun.sh"
          rm temprun.sh
        done
    done
  done
done

for method in ${methods[@]}
do
  for env in ${envname[@]}
  do
    for seed in ${seeds[@]}
    do
      echo "#!/bin/bash" >> temprun.sh
      echo "#SBATCH --account=def-dprecup" >> temprun.sh
      echo "#SBATCH --output=\"/scratch/syarnob/DR_slurm/slurm-%j.out\"" >> temprun.sh
      echo "#SBATCH --job-name=DREst_env_${env}_method_${method}" >> temprun.sh
      echo "#SBATCH --cpus-per-task=6"  >> temprun.sh
      echo "#SBATCH --gres=gpu:1" >> temprun.sh
      echo "#SBATCH --mem=10G" >> temprun.sh
      echo "#SBATCH --time=12:00:00" >> temprun.sh
      echo "module load singularity" >> temprun.sh
      echo "echo "Syncing singularity image..." " >> temprun.sh
      echo "cp -r /scratch/syarnob/rllab.simg \$SLURM_TMPDIR/" >> temprun.sh
      echo "echo "Running image." " >> temprun.sh
      echo "singularity exec --nv -B \$SLURM_TMPDIR:/tmp \$SLURM_TMPDIR/rllab.simg python main.py --activate_HDR --policy_name ${method[@]} --env_name ${env[@]} --seed ${seed[@]}" >> temprun.sh
              cat temprun.sh
              eval "sbatch temprun.sh"
      rm temprun.sh
    done
  done
done

