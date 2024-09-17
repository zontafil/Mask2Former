#!/bin/bash
#SBATCH --job-name=SwinT_Maxvector
#SBATCH --account=project_462000238
#SBATCH --partition=standard-g
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --output=../slurm_run_logs/Train_SwinT_instance_1node_part2.txt

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

# Python environment
cd ..
source myenv/bin/activate

# Set up AMD environment
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-detectron"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# Run instance segmentation training on multi-gpu and multi-node
#srun python3 train_net.py --num-gpus 8 --num-machines 1 --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml MODEL.WEIGHTS checkpoints/swin_tiny_patch4_window7_224.pkl

# Resume training. Resume does not need weight spcified, instead it used weights specified: ./experiments/SwinT/last_checkpoint
srun python3 train_net.py --num-gpus 8 --num-machines 1 --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml --resume