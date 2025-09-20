#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 71:20:00
#SBATCH --account=loni_quantum04
#SBATCH --output=logs/convert_cnf_to_3D_packing_method_deepseek_reasoner_output.log
#SBATCH --error=logs/convert_cnf_to_3D_packing_method_deepseek_reasoner_error.log




module purge
module load cuda/11.8.0
export CUDA_HOME=/usr/local/packages/cuda/11.8.0/7oedjh4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate Satisfiability_Solvers
python -c "import sys; print(sys.executable)"  # 确认 Python 路径



python convert_cnf_to_3D_packing_method_deepseek_reasoner.py


conda deactivate

