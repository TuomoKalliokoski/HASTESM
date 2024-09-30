import logging
import subprocess
import time
from pathlib import Path
from typing import List, Union

from configargparse import ArgumentError


def run_command(command: List[Union[str, Path]], error_message: str, logger: logging.Logger) -> str:
	"""Execute command using subprocess.run.

	Args:
	----
		command: command to execute
		error_message: message to print in case of error
		logger: logging.Logger from the main script

	Returns:
	-------
		stdout of the command

	"""
	command = [str(c) for c in command]

	try:
		result = subprocess.run(command, check=True, text=True, capture_output=True)
	except subprocess.CalledProcessError as e:
		logger.error(f"{error_message}. Command: {' '.join(command)}")
		logger.error(f'Return code: {e.returncode}')
		logger.error(f'stdout: {e.stdout}')
		logger.error(f'stderr: {e.stderr}')
		raise
	else:
		logger.debug(f"Command: {' '.join(command)}")
		logger.debug(f'{result.returncode}: {result.stdout}')
		return result.stdout


def submit_slurm_job(script: str, logger: logging.Logger) -> str:
	command = ['sbatch', '--parsable', script]
	jobid = run_command(command, 'Failed to submit Slurm job', logger).strip()
	return jobid


def wait_for_job(jobid: str, job_ids: List[str]) -> None:
	while True:
		command = ['squeue', '-h', '-j', jobid, '-o', '%i']
		result = subprocess.run(command, text=True, capture_output=True).stdout
		if not result:
			try:
				job_ids.remove(jobid)
			except ValueError:
				pass
			break

		time.sleep(30)


def existing_file(path_string: str) -> Path:
	path = Path(path_string).resolve()
	if not path.is_file():
		raise FileNotFoundError(f"'{path}' is not a file")

	return path


def new_dir(path_string: str) -> Path:
	path = Path(path_string).resolve()

	if path.is_file():
		raise ArgumentError(f"'{path}' is not a directory")

	path.mkdir(parents=True, exist_ok=True)

	return path


def any_path(path_string: str) -> Path:
	return Path(path_string).resolve()
