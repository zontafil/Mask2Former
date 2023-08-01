#!/bin/bash
#SBATCH --job-name=SwinS_evaluate
#SBATCH --account=project_462000238
#SBATCH --partition=small-g
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --output=../slurm_run_logs/Evaluate_SwinS.txt

cd ..
source myenv/bin/activate
srun python3 train_net.py --num-gpus 1 --eval-only --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml MODEL.WEIGHTS checkpoints/coco_instance_Swin-S.pkl