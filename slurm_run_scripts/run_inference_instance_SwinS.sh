#!/bin/bash
#SBATCH --job-name=SwinS_inference
#SBATCH --account=project_462000238
#SBATCH --partition=small-g
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --output=../slurm_run_logs/Inference_SwinS.txt

srun python3 ../demo/demo.py --config-file ../configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml --input ../datasets/test_inputs/img01.jpg --output ../datasets/test_outputs/ --opts MODEL.WEIGHTS ../checkpoints/coco_instance_Swin-S.pkl