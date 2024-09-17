#!/bin/bash
#SBATCH --job-name=SwinT_Train
#SBATCH --account=project_2006531
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --mail-type=END
#SBATCH --output=../csc-run-logs/Train_SwinT_panoptic_15min.txt

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

# Run instance segmentation training on multi-gpu and multi-node
srun python3 ../train_net.py --num-gpus 2 --num-machines 1 --config-file configs/coco/panoptic-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml MODEL.WEIGHTS checkpoints/swin_tiny_patch4_window7_224.pkl

# Resume training. Resume does not need weight spcified, instead it used weights specified: ./experiments/SwinT/last_checkpoint
#srun python3 ../train_net.py --num-gpus 2 --num-machines 1 --config-file configs/coco/panoptoc-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml --resume