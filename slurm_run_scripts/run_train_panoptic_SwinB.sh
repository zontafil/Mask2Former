#!/bin/bash
#SBATCH --job-name=SwinB_Train
#SBATCH --account=project_2006531
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-type=END
#SBATCH --output=../csc-run-logs/Train_panoptic_SwinB.txt

### Define master port and set the first node name as master address
# https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp
export MASTER_PORT=29400
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

##### Print the SLURM job info
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "NODELIST: " $SLURM_JOB_NODELIST
echo "MASTER_ADDR: "$MASTER_ADDR
echo "Number of nodes: " $SLURM_JOB_NUM_NODES
echo "Ntasks per node: "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# Load Pytorch module
module load pytorch/1.13

# Instance segmentation training
#srun python3 ../train_net.py --num-gpus 4 --num-machines 2 --config-file configs/coco/panoptic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml MODEL.WEIGHTS checkpoints/swin_base_patch4_window12_384_22k.pkl

# # Resume training. Resume does not need weight spcified, instead it used weights specified: ./experiments/SwinB/last_checkpoint
srun python3 ../train_net.py --num-gpus 4 --num-machines 2 --config-file configs/coco/panoptic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml --resume