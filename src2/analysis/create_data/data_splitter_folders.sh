#!/bin/bash
#SBATCH --mem=100g
#SBATCH -c24
#SBATCH --time=1-0

#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=gringolet-01,gringolet-02,gringolet-03,gringolet-04,gringolet-05,gringolet-06


export PYTHONPATH=/cs/labs/gabis/eliyahabba/LLM-Evaluation/

sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
dir="/cs/labs/gabis/eliyahabba/LLM-Evaluation/src/analysis/create_data"
absolute_path=$(readlink -f $dir)
# print the full (not relative) path of the dir variable
echo "current dir is set to: $absolute_path"
cd $dir

echo ${SLURM_ARRAY_TASK_ID}
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
python DatasetSplitter.py; # input="ibm_results_data_full_processed", output="ibm_results_data_full_processed_split"
python DatasetSplitterDeDupAllCols.py # input="ibm_results_data_full_processed_split", output="ibm_results_data_full_processed_split_all_cols_deduped"
CUDA_LAUNCH_BLOCKING=1 python DatasetSplitterFolders.py; # input="/ibm_results_data_full_processed_split_all_cols_deduped", output="ibm_results_data_full_processed_split_to_folders"
python DatasetSplitterFoldersDeDupAllCols.py ; # input="/ibm_results_data_full_processed_split_to_folders", output="ibm_results_data_full_processed_split_to_folders_all_cols_deduped"
python upload_data_for_analysis_split_folders.py;
python upload_data_for_analysis_split_folders_private.py
