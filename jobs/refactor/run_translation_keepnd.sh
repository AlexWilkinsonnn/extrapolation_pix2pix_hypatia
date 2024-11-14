#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 1440
#SBATCH -J translate
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s"
#SBATCH --array=1-134
#SBATCH --error=/home/awilkins/extrapolation_pix2pix/jobs/logs/err/%x.%A_%a.err
#SBATCH --output=/home/awilkins/extrapolation_pix2pix/jobs/logs/out/%x.%A_%a.out

################################################################################
# Options

SCRATCH_DIR="/state/partition1/awilkins/scratch/${SLURM_JOB_ID}"

INPUT_DIR=$1
OUTPUT_DIR=$2

################################################################################

mkdir -p ${SCRATCH_DIR}

input_name=$(ls $INPUT_DIR | head -n $SLURM_ARRAY_TASK_ID | tail -n -1)
input_file=${INPUT_DIR}/${input_name}
output_name=${input_name%.*}_fdresppred.h5
output_file=${SCRATCH_DIR}/${output_name}

echo "Job id ${SLURM_JOB_ID}"
echo "Job array task id ${SLURM_ARRAY_TASK_ID}"
echo "Running on ${SLURM_JOB_NODELIST}"
echo "With cuda device ${CUDA_VISIBLE_DEVICES}"
echo "Input file is ${input_file}"
echo "Output file is ${output}"
echo "Output dir is ${OUTPUT_DIR}"

cd /home/awilkins/extrapolation_pix2pix
source setups/setup.sh

python run_translation.py --signalmask_max_tick_positive_Z 30 \
                          --signalmask_max_tick_negative_Z 30 \
                          --signalmask_max_ch_Z 4 \
                          --signalmask_max_tick_positive_U 100 \
                          --signalmask_max_tick_negative_U 75 \
                          --signalmask_max_ch_U 8 \
                          --signalmask_max_tick_positive_V 100 \
                          --signalmask_max_tick_negative_V 75 \
                          --signalmask_max_ch_V 8 \
                          --threshold_mask_Z 10 \
                          --threshold_mask_U 10 \
                          --threshold_mask_V 10 \
                          --add_noise_from bin/FDPlanesNoiseExample.h5 \
                          ${input_file} \
                          ${output_file} \
                          checkpoints/thesis_infilled_6chs_minadc100_Z/exp26_continue1decay1final/config.yaml \
                          best_bias_mu \
                          checkpoints/thesis_infilled_6chs_minadc100_largersigmask_U/exp1_continuedecay2final/config.yaml \
                          best_loss_channel \
                          checkpoints/thesis_infilled_6chs_minadc100_largersigmask_V/exp1_continuedecay2final/config.yaml \
                          best_loss_pix


if [[ $? == 0 ]]
then
  cp ${output_file} ${OUTPUT_DIR}/${output_name}
else
  echo "Python script exited badly for file ${input_name}!"
fi

rm -r ${SCRATCH_DIR}
