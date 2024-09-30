import csv
import sqlite3
import tarfile
from logging import Logger
from pathlib import Path
from typing import List

from ..generate_slurm_scripts import generate_predict_script
from ..hastesm_types import HastesmParams
from ..utils import submit_slurm_job, wait_for_job


def decompress_predictions(output_dir: Path, local_dir: Path, pred_cpu: int, logger: Logger) -> None:
	"""Decompress predictions

	Args:
	----
	output_dir: Path to the output directory
	local_dir: Path to the local directory
	pred_cpu: Number of prediction CPUs
	logger: logging.Logger instance

	"""
	for i in range(1, pred_cpu + 1):
		pred_tar = output_dir / f'PRED{i}.tar.gz'

		if not pred_tar.exists():
			logger.warning(f'Warning: {pred_tar} not found in {output_dir}')
			continue

		logger.debug(f'Decompressing {pred_tar}...')
		with tarfile.open(pred_tar, 'r:gz') as tar:
			tar.extractall(path=local_dir)

		pred_tar.unlink()


def import_pred(output_dir: Path, local_dir: Path, pred_db: Path, pred_cpu: int, logger: Logger) -> None:
	"""Unzip and import chemprop predictions to pred_db

	Args:
	----
	output_dir: Path to the output directory
	local_dir: Path to the local directory
	pred_db: Path to the HASTESM prediction database
	pred_cpu: Number of prediction CPUs
	logger: logging.Logger instance

	"""
	logger.info('Decompressing predictions')
	decompress_predictions(output_dir, local_dir, pred_cpu, logger)

	# Equivalent to: rm -f pred_db
	pred_db.unlink(missing_ok=True)

	conn = sqlite3.connect(pred_db)
	c = conn.cursor()
	c.execute('CREATE TABLE pred_data (hastenid INTEGER PRIMARY KEY, pred_score NUMERIC)')

	logger.info(f'Importing predictions to {pred_db}...')

	predfiles = list(local_dir.glob('PRED*/output_pred_chunk*.csv'))
	num_preds = 0
	for i, filename in enumerate(predfiles):
		logger.debug(f'Processing: {filename} [{i+1} of {len(predfiles)}]')

		with open(filename) as smilesfile:
			reader = csv.DictReader(smilesfile, delimiter=',')

			c.executemany(
				'INSERT INTO pred_data(hastenid, pred_score) VALUES (?,?)',
				((int(row['hastenid']), float(row['similarity'])) for row in reader),
			)

		conn.commit()

		num_preds += c.rowcount

	logger.info(f'Imported {num_preds} predictions')

	conn.close()


def predict(params: HastesmParams, logger: Logger, job_ids: List[str]) -> None:
	"""Predicts similarities of the molecules in the HASTESM database to the query with the chemprop model

	Args:
	----
		params: Parsed HastesmParams from the main file
		logger: logging.Logger instance
		job_ids: List of slurm job ids where the predict job id is added

	Raises:
	------
		ValueError: If the model_path does not exist

	"""
	if not params.model_path.exists():
		msg = f'{params.model_path} does not exist'
		logger.error(msg)
		raise ValueError(msg)

	script_path = generate_predict_script(
		params.name,
		params.output_dir,
		params.local_dir,
		params.cpu_partition,
		params.pred_cpu,
		params.model_path,
		params.debug,
	)

	predict_jobid = submit_slurm_job(script_path, logger)
	job_ids.append(predict_jobid)

	logger.info(f'Submitted chemprop predicting job with ID {predict_jobid}')

	wait_for_job(predict_jobid, job_ids)
	logger.info(f'Predicting job {predict_jobid} completed')

	# import predictions
	import_pred(params.output_dir, params.local_dir, params.pred_db, params.pred_cpu, logger)
