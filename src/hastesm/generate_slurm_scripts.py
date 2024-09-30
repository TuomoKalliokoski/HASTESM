from pathlib import Path
from typing import List, Optional


def generate_confgen_script(
	name: str,
	output_dir: Path,
	local_dir: Path,
	cpu_partition: str,
	conf_cpu: int,
	iteration: int,
	logging_enabled=False,
) -> Path:
	"""Generate SLURM script for conformer generation.

	Args:
	----
		name: Name of the job
		output_dir: Path to the output directory
		local_dir: Path to the local directory
		cpu_partition: SLURM partition to use for CPU computations
		conf_cpu: Number of confgen CPUs
		iteration: Shape matching iteration number
		logging_enabled: Whether to enable logging

	Returns:
	-------
		Path to the generated script

	"""
	script_path = output_dir / f'{name}_confgen.sh'

	if logging_enabled:
		logs_dir = output_dir / 'logs'
		logs_dir.mkdir(exist_ok=True)
		out = logs_dir / 'confgen_out_%A_%a.txt'
		err = logs_dir / 'confgen_err_%A_%a.txt'
	else:
		out = '/dev/null'
		err = '/dev/null'


	script = f"""#!/bin/bash
#SBATCH -J hastesm_confgen
#SBATCH -o {out}
#SBATCH -e {err}
#SBATCH -p {cpu_partition}
#SBATCH --cpus-per-task=1
#SBATCH --array=1-{conf_cpu}

set -u
WORK_DIR={local_dir}/$SLURM_JOB_ID/confgen_cpu$SLURM_ARRAY_TASK_ID
rm -rf $WORK_DIR
mkdir -p $WORK_DIR
cd $WORK_DIR
cp {output_dir}/{name}_iter{iteration}_cpu$SLURM_ARRAY_TASK_ID.smi $WORK_DIR/
cp {output_dir}/{name}_iter{iteration}_cpu$SLURM_ARRAY_TASK_ID.inp $WORK_DIR/
export SCHRODINGER_FEATURE_FLAGS=""
$SCHRODINGER/pipeline \
	-prog phase_db {name}_iter{iteration}_cpu$SLURM_ARRAY_TASK_ID.inp \
	-OVERWRITE \
	-WAIT \
	-HOST localhost:1 \
	-NJOBS 1

tar --exclude=confs-{name}_iter{iteration}_cpu$SLURM_ARRAY_TASK_ID.tar.gz \
	-czf confs-{name}_iter{iteration}_cpu$SLURM_ARRAY_TASK_ID.tar.gz .

mv confs-{name}_iter{iteration}_cpu$SLURM_ARRAY_TASK_ID.tar.gz {output_dir}/
rm -rf $WORK_DIR
"""

	with open(script_path, 'w') as f:
		f.write(script)

	return script_path


def generate_align_script(
	name: str,
	output_dir: Path,
	local_dir: Path,
	query: Path,
	cpu_partition: str,
	conf_cpu: int,
	search_cpu: int,
	iteration: int,
	logging_enabled=False,
) -> Path:
	"""Generate SLURM script for aligning conformers with SCHRODINGER shape_screen.

	Args:
	----
		name: Name of the job
		output_dir: Path to the output directory
		local_dir: Path to the local directory
		query: Path to the query SDF file
		cpu_partition: SLURM partition to use for CPU computations
		conf_cpu: Number of confgen CPUs
		search_cpu: Number of search CPUs
		iteration: Shape matching iteration number
		logging_enabled: Whether to enable logging

	Returns:
	-------
		Path to the generated script

	"""
	script_path = output_dir / f'{name}_align.sh'

	if logging_enabled:
		logs_dir = output_dir / 'logs'
		logs_dir.mkdir(exist_ok=True)
		out = logs_dir / 'align_out_%A_%a.txt'
		err = logs_dir / 'align_err_%A_%a.txt'
	else:
		out = '/dev/null'
		err = '/dev/null'


	script = f"""#!/bin/bash
#SBATCH -J hastesm_align
#SBATCH -o {out}
#SBATCH -e {err}
#SBATCH -p {cpu_partition}
#SBATCH --cpus-per-task=1
#SBATCH --array=1-{conf_cpu}%{search_cpu}

set -u
WORK_DIR={local_dir}/$SLURM_JOB_ID/align_cpu$SLURM_ARRAY_TASK_ID
rm -rf $WORK_DIR
mkdir -p $WORK_DIR
cd $WORK_DIR
tar xzf {output_dir}/confs-{name}_iter{iteration}_cpu$SLURM_ARRAY_TASK_ID.tar.gz \
	-C $WORK_DIR/

cp {query} $WORK_DIR/query.sdf
export SCHRODINGER_FEATURE_FLAGS=""
$SCHRODINGER/shape_screen \
	-shape $WORK_DIR/query.sdf \
	-screen `pwd`/{name}_iter{iteration}_cpu$SLURM_ARRAY_TASK_ID.phdb \
	-JOB {name}_iter{iteration}_cpu$SLURM_ARRAY_TASK_ID \
	-norm 1 \
	-HOST localhost:1 \
	-WAIT

mv {name}_iter{iteration}_cpu${{SLURM_ARRAY_TASK_ID}}_align.maegz {output_dir}/
rm -rf $WORK_DIR
"""

	with open(script_path, 'w') as f:
		f.write(script)

	return script_path


def generate_train_script(
	name: str,
	output_dir: Path,
	local_dir: Path,
	model_path: Path,
	gpu_partition: str,
	init_conda: str,
	activate_conda: str,
	logging_enabled=False,
) -> Path:
	"""Generate SLURM script for training with chemprop.

	Args:
	----
		name: Name of the job
		output_dir: Path to the output directory
		local_dir: Path to the local directory
		model_path: Path where the trained chemprop model should be saved
		gpu_partition: SLURM partition to use for GPU computations
		init_conda: Command to initialize conda
		activate_conda: Command to activate the conda environment
		logging_enabled: Whether to enable logging

	Returns:
	-------
		Path to the generated script

	"""
	script_path = output_dir / f'{name}_train.sh'

	if logging_enabled:
		logs_dir = output_dir / 'logs'
		logs_dir.mkdir(exist_ok=True)
		out = logs_dir / 'train_out_%j.txt'
		err = logs_dir / 'train_err_%j.txt'
	else:
		out = '/dev/null'
		err = '/dev/null'

	script = f"""#!/bin/bash
#SBATCH -J hastesm_train
#SBATCH -o {out}
#SBATCH -e {err}
#SBATCH -p {gpu_partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive

set -u
{init_conda}
{activate_conda}
WORK_DIR={local_dir}/$SLURM_JOB_ID/train
rm -rf $WORK_DIR
mkdir -p $WORK_DIR
cd $WORK_DIR
cp {output_dir}/train_{name}.csv.gz $WORK_DIR/
gunzip $WORK_DIR/train_{name}.csv.gz

OMP_NUMBER_THREADS=1 chemprop train \
	--num-workers 0 \
	--task-type regression \
	--accelerator gpu \
	--devices 1 \
	--smiles-columns smiles \
	--target-columns similarity \
	--data-path $WORK_DIR/train_{name}.csv \
	--save-dir {model_path} \
	--batch-size 256 \
	--epochs 20

rm -rf $WORK_DIR
"""

	with open(script_path, 'w') as f:
		f.write(script)

	return script_path


def generate_predict_script(
	name: str,
	output_dir: Path,
	local_dir: Path,
	cpu_partition: str,
	pred_cpu: int,
	model_path: Path,
	logging_enabled=False,
) -> Path:
	"""Generate SLURM script for prediction with chemprop.

	Args:
	----
		name: Name of the job
		iteration: HASTESM iteration number
		output_dir: Path to the output directory
		local_dir: Path to the local directory
		cpu_partition: SLURM partition to use for CPU computations
		pred_cpu: Number of prediction CPUs
		model_path: Path to the trained chemprop model
		logging_enabled: Whether to enable logging

	Returns:
	-------
		Path to the generated script

	"""
	script_path = output_dir / f'{name}_predict.sh'

	if logging_enabled:
		logs_dir = output_dir / 'logs'
		logs_dir.mkdir(exist_ok=True)
		out = logs_dir / 'predict_out_%A_%a.txt'
		err = logs_dir / 'predict_err_%A_%a.txt'
	else:
		out = '/dev/null'
		err = '/dev/null'

	script = f"""#!/bin/bash
#SBATCH -J hastesm_predict
#SBATCH -o {out}
#SBATCH -e {err}
#SBATCH -p {cpu_partition}
#SBATCH --cpus-per-task=1
#SBATCH--array=1-{pred_cpu}

set -u
WORK_DIR={local_dir}/$SLURM_JOB_ID/predict_cpu$SLURM_ARRAY_TASK_ID
rm -rf $WORK_DIR/PRED$SLURM_ARRAY_TASK_ID
mkdir -p $WORK_DIR/PRED$SLURM_ARRAY_TASK_ID
cd $WORK_DIR/
cp -r {model_path}/ $WORK_DIR/
mv {output_dir}/pred_{name}_cpu$SLURM_ARRAY_TASK_ID.sh $WORK_DIR/
source pred_{name}_cpu$SLURM_ARRAY_TASK_ID.sh
tar -czf {output_dir}/PRED$SLURM_ARRAY_TASK_ID.tar.gz \
	PRED$SLURM_ARRAY_TASK_ID

rm -rf $WORK_DIR
"""

	with open(script_path, 'w') as f:
		f.write(script)

	return script_path
