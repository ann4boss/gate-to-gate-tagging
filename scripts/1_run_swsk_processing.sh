#!/bin/bash
#SBATCH --job-name=run_swsk_processing
#SBATCH --partition=pibu_el8
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/users/egraf/deep_learning/logs/%x_%j.out
#SBATCH --error=/data/users/egraf/deep_learning/logs/%x_%j.err

#*-----User-editable variables-----
WORKDIR="/data/users/${USER}/deep_learning"
OUTDIR="${WORKDIR}/data/frames"


#*-----Create Ouput directory-----
mkdir -p "$OUTDIR"

cd "$WORKDIR"

#*-----Load modules-----
module add Python 

#*-----Activate virtual environment-----
source "${WORKDIR}/venv/bin/activate"

#*----- Check things look right -----
echo "Working dir : $(pwd)"
echo "Python      : $(which python)"
echo "Python ver  : $(python --version)"


#*-----Run Python script-----
python scripts/swsk_process_videos.py 