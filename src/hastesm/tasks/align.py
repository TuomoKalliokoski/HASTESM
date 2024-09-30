import os
from importlib.resources import path as pkg_path
from logging import Logger
from pathlib import Path
from typing import List

from ..generate_slurm_scripts import generate_align_script
from ..hastesm_types import HastesmParams
from ..utils import run_command, submit_slurm_job, wait_for_job


def import_search(
	name: str,
	output_dir: Path,
	db: Path,
	search_db: Path,
	matches_db: Path,
	cutoff: float,
	iteration: int,
	logger: Logger,
) -> None:
	"""Runs a separate python script in the SCHRODINGER environment"""
	if not list(output_dir.glob(f'{name}_iter{iteration}_cpu*_align.maegz')):
		msg = f'No shape matching results found in {output_dir}'
		logger.critical(msg)
		raise ValueError(msg)

	schrodinger_path = os.environ.get('SCHRODINGER')
	if not schrodinger_path:
		raise EnvironmentError('Error: SCHRODINGER environment variable not set')

	with pkg_path('hastesm', 'import_search.py') as script_path:
		run_command(
			[
				f'{schrodinger_path}/run',
				script_path,
				'--name', name,
				'--db', db,
				'--output-dir', output_dir,
				'--search-db', search_db,
				'--matches-db', matches_db,
				'--cutoff', str(cutoff),
				'--iteration', str(iteration),
			],
			'Error importing shapematching results',
			logger,
		)


def align(params: HastesmParams, logger: Logger, job_ids: List[str], iteration: int) -> None:
	"""Computes similarity between query and generated conformers using SCHRODINGER shape_screen

	Args:
	----
		params: Parsed HastesmParams from the main file
		logger: logging.Logger instance
		job_ids: List of slurm job ids where the align job id is added
		iteration: Shape matching iteration number (1 or 2)

	"""
	align_script_path = generate_align_script(
		params.name,
		params.output_dir,
		params.local_dir,
		params.query,
		params.cpu_partition,
		params.conf_cpu,
		params.search_cpu,
		iteration,
		params.debug,
	)
	align_jobid = submit_slurm_job(align_script_path, logger)
	job_ids.append(align_jobid)

	logger.info(f'Submitting align job with ID {align_jobid}')

	wait_for_job(align_jobid, job_ids)

	logger.info(f'Align job {align_jobid} completed')
	logger.info(f'Importing results to {params.search_db}')

	import_search(
		params.name, params.output_dir, params.db, params.search_db, params.matches_db, params.cutoff, iteration, logger
	)

	inp_files = params.output_dir.glob(f'{params.name}_iter{iteration}_cpu*.inp')
	smi_files = params.output_dir.glob(f'{params.name}_iter{iteration}_cpu*.smi')
	shp_files = params.output_dir.glob(f'shapematching_{params.name}_iter{iteration}_cpu*.sh')
	conf_files = params.output_dir.glob(f'confs-{params.name}_iter{iteration}_cpu*.tar.gz')

	for inp_file in inp_files:
		inp_file.unlink()
	for smi_file in smi_files:
		smi_file.unlink()
	for shp_file in shp_files:
		shp_file.unlink()
	for conf_file in conf_files:
		conf_file.unlink()
