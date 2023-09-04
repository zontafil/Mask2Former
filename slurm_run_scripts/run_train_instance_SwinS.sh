#!/bin/bash
#SBATCH --job-name=SwinS_Maxvector
#SBATCH --account=project_462000238
#SBATCH --partition=standard-g
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=63
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --output=../slurm_run_logs/Train_SwinS_instance.txt

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

###
cd ..
source myenv/bin/activate

export MIOPEN_USER_DB_PATH="/tmp/my-miopen-detectron"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# Run instance segmentation training on multi-gpu and multi-node
srun python3 train_net.py --num-gpus 4 --num-machines 2 --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml MODEL.WEIGHTS checkpoints/swin_small_patch4_window7_224.pkl
#srun python3 ../train_net.py --num-gpus 4 --num-machines 2 --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml MODEL.WEIGHTS experiments/coco_instance_SwinS/model_final.pth

# Resume training. Resume does not need weight spcified, instead it used weights specified: ./experiments/SwinS/last_checkpoint
#run python3 ../train_net.py --num-gpus 4 --num-machines 2 --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml --resume