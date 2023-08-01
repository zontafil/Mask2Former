#!/bin/bash
#SBATCH --job-name=R101_Maxvector
#SBATCH --account=project_2006531
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-type=END
#SBATCH --output=../csc-run-logs/Train_R101_instance_new_augs_part2.txt

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
#srun python3 ../train_net.py --num-gpus 4 --num-machines 2 --config-file configs/coco/instance-segmentation/maskformer2_R101_bs16_50ep.yaml MODEL.WEIGHTS checkpoints/R-101.pkl
#srun python3 ../train_net.py --num-gpus 4 --num-machines 2 --config-file configs/coco/instance-segmentation/maskformer2_R101_bs16_50ep.yaml MODEL.WEIGHTS experiments/coco_instance_R101/model_final.pth

# Resume training. Resume does not need weight spcified, instead it used weights specified: ./experiments/R101/last_checkpoint
srun python3 ../train_net.py --num-gpus 4 --num-machines 1 --config-file configs/coco/instance-segmentation/maskformer2_R101_bs16_50ep.yaml --resume