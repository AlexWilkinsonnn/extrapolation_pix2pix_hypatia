#!/bin/bash
#SBATCH -p RCIF
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -J make_infill_pairs
#SBATCH --array=1-180
#SBATCH --error=/home/awilkins/extrapolation_pix2pix/jobs/logs/err/%x.%A_%a.err
#SBATCH --output=/home/awilkins/extrapolation_pix2pix/jobs/logs/out/%x.%A_%a.out

################################################################################
# Options

INPUT_DIR=$1
OUTPUT_DIR=$2
SIGNAL_TYPE=$3
MIN_ADC=$4
SIGMASK_MAX_TICK=$5
SIGMASK_MAX_CH=$5

START_IDX_STEP=100000

################################################################################

start_idx=$(($START_IDX_STEP * $SLURM_ARRAY_TASK_ID))
input_name=$(ls $INPUT_DIR | head -n $SLURM_ARRAY_TASK_ID | tail -n -1)
input_file=${INPUT_DIR}/${input_name}

echo "Job id ${SLURM_JOB_ID}"
echo "Job array task id ${SLURM_ARRAY_TASK_ID}"
echo "Running on ${SLURM_JOB_NODELIST}"
echo "With cuda device ${CUDA_VISIBLE_DEVICES}"
echo "Input file is ${input_file}"
echo "Start index is ${start_idx}"
echo "Output dir is ${OUTPUT_DIR}"
echo "Signal type is ${SIGNAL_TYPE}"
echo "Min ADC is ${MIN_ADC}"

cd /home/awilkins/extrapolation_pix2pix
source setups/setup.sh

python data_scripts/make_infilled_pairs.py --min_adc $MIN_ADC \
                                           --start_idx $start_idx \
                                           --signalmask_max_tick $SIGMASK_MAX_TICK \
                                           --signalmask_max_ch $SIGMASK_MAX_CH \
                                           --batch_mode \
                                           $input_file \
                                           $OUTPUT_DIR \
                                           $SIGNAL_TYPE


if [[ $? != 0 ]]
then
  echo "Python script exited badly for file ${input_name}!"
fi
