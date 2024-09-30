import atexit
import logging
import shutil
import sys
from enum import Enum, auto
from functools import partial
from importlib.resources import path as pkg_path
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple, Union

from configargparse import ArgumentParser

from .hastesm_types import ConfgenParams, HastesmParams, TaskName
from .tasks.align import align
from .tasks.confgen import confgen
from .tasks.export import export
from .tasks.predict import predict
from .tasks.prepare_db import prepare_db
from .tasks.status import status
from .tasks.train import train
from .utils import any_path, existing_file, new_dir, run_command

FIRST_TASK = 'prepare_db'
NUM_TASKS = len(list(TaskName))


def next_task_name(task_name: str) -> Union[TaskName, None]:
	"""Get the next task as a TaskName

	Args:
	----
	task_name: Name of the current task as a string

	Raises:
	------
	ValueError: If the task_name is not a valid TaskName

	Returns:
	-------
		Union[TaskName, None]: TaskName of the next task, or None if there is no next task

	"""
	if task_name not in TaskName:
		raise ValueError(f'Invalid task name: {task_name}')

	task = TaskName(task_name)
	task_list = list(TaskName)

	current_index = task_list.index(task)
	return task_list[current_index + 1] if current_index + 1 < len(task_list) else None


def get_name_of_task_dependency(task_name: str) -> Union[TaskName, None]:
	"""Get the TaskName of the task that the task depends on

	Args:
	----
		task_name: Name of the task as a string

	Raises:
	------
		ValueError: If the task_name is not a valid TaskName

	Returns:
	-------
		Union[TaskName, None]: TaskName of the task that the task depends on, or None if the task has no dependency

	"""
	if task_name not in TaskName:
		raise ValueError(f'Invalid task name: {task_name}')

	if task_name == 'status':
		return TaskName.ALIGN_2

	task = TaskName(task_name)
	task_list: List[TaskName] = list(TaskName)

	current_index = task_list.index(task)
	return task_list[current_index - 1] if current_index - 1 >= 0 else None


class TaskState(Enum):

	"""State of a task"""

	PENDING = auto()
	RUNNING = auto()
	COMPLETED = auto()


class Task:

	"""HASTESM task used for checkpointing the workflow"""

	def __init__(self, task_name: TaskName, func: Callable, logger: logging.Logger, confgen_task: bool = False) -> None:
		self.name = task_name
		self.func = func
		self.state = TaskState.PENDING
		self.logger = logger
		self.next: Optional[Task] = None
		self.confgen_task = confgen_task

	def __call__(
		self, params: HastesmParams, job_ids: List[str], confgen_params: Optional[ConfgenParams] = None
	) -> None:
		self.state = TaskState.RUNNING
		try:
			if self.confgen_task:
				if confgen_params is None:
					msg = f'confgen parameters must be provided for {self.name}'
					self.logger.error(msg)
					raise ValueError(msg)
				self.func(params, confgen_params, self.logger, job_ids)
			else:
				self.func(params, self.logger, job_ids)

			self.state = TaskState.COMPLETED
			self.logger.info(f'Task {self.name} completed successfully.')
		except Exception as e:
			self.logger.critical(f'Task {self.name} failed with error: {e}')
			raise


class TaskQueue:

	"""Defines the HASTESM workflow, a linked list of tasks"""

	def __init__(self, logger: logging.Logger):
		"""Create the linked list of tasks

		Args:
		----
			logger (logging.Logger): logging.Logger instance

		"""
		self.head = self._create_task_list(logger)
		self.current: Union[Task, None] = self.head
		self.job_ids: List[str] = []

	def _create_task_list(self, logger: logging.Logger) -> Task:
		prepare_db_task = Task(TaskName.PREPARE_DB, prepare_db, logger)
		confgen_1_task = Task(TaskName.CONFGEN_1, partial(confgen, iteration=1), logger, confgen_task=True)
		align_1_task = Task(TaskName.ALIGN_1, partial(align, iteration=1), logger)
		train_task = Task(TaskName.TRAIN, train, logger)
		predict_task = Task(TaskName.PREDICT, predict, logger)
		confgen_2_task = Task(TaskName.CONFGEN_2, partial(confgen, iteration=2), logger, confgen_task=True)
		align_2_task = Task(TaskName.ALIGN_2, partial(align, iteration=2), logger)
		export_task = Task(TaskName.EXPORT, export, logger)
		status_task = Task(TaskName.STATUS, status, logger)

		prepare_db_task.next = confgen_1_task
		confgen_1_task.next = align_1_task
		align_1_task.next = train_task
		train_task.next = predict_task
		predict_task.next = confgen_2_task
		confgen_2_task.next = align_2_task
		align_2_task.next = export_task
		export_task.next = status_task

		return prepare_db_task

	def start_from(self, task_name: str) -> None:
		"""Start the workflow from the specified task

		Args:
		----
			task_name: Name of the task to start from as a string

		Raises:
		------
			ValueError: If the task_name is not a valid TaskName

		"""
		task: Union[Task, None] = self.head
		while task is not None and task.name != task_name:
			task = task.next
		if task is not None:
			self.current = task
		else:
			raise ValueError(f'Task {task_name} not found in the queue')

	def run_next_task(self, params: HastesmParams, confgen_params: Optional[ConfgenParams] = None) -> Union[Task, None]:
		"""Run the next task in the workflow

		Args:
		----
			params: Parsed HastesmParams from the main file
			confgen_params: Parsed arguments for conformer generation. Defaults to None.

		Returns:
		-------
			Union[Task, None]: Completed task or None if there is no next task

		"""
		if self.current is not None:
			self.current(params, self.job_ids, confgen_params)
			completed_task = self.current
			self.current = self.current.next
			return completed_task
		return None

	def has_next(self) -> bool:
		return self.current is not None

	def cancel_all_jobs(self, logger: logging.Logger) -> None:
		"""Cancel all jobs in the queue

		Args:
		----
			logger: logging.Logger instance

		"""
		for job_id in self.job_ids:
			logger.debug(f'Canceling job {job_id}')
			run_command(['scancel', '--quiet', job_id], 'Failed to cancel job', logger)
		self.job_ids.clear()


def parse_arguments() -> Tuple[HastesmParams, ConfgenParams, str]:
	with pkg_path('hastesm', 'default_config.txt') as config_path:
		parser = ArgumentParser(default_config_files=[str(config_path), './*.conf', './config.txt'])

		parser.add_argument(
			'--debug',
			default=False,
			action='store_true',
			help='Log debugging information such as SLURM job stdout and stderr',
		)

		parser.add('-c', '--my-config', is_config_file=True, help='config file path')
		parser.add_argument('--init-conda', required=True, type=str, help='Command to initialize conda')
		parser.add_argument(
			'--activate-conda', required=True, type=str, help='Command to activate the chemprop conda environment'
		)
		parser.add_argument(
			'--mae-format-version', required=True, type=str, help='Maestro file format version for exporting matches'
		)

		parser.add_argument('--name', required=True, type=str, help='Unique name for the search')
		parser.add_argument(
			'--output-dir', required=True, type=new_dir, help='Path to the output directory on the NFS drive'
		)
		parser.add_argument(
			'--local-dir', required=True, type=new_dir, help='Path to the output directory on the local drive'
		)
		parser.add_argument('--db', required=True, type=any_path, help='Path to the HASTESM database')
		parser.add_argument(
			'--input-smi', type=any_path, help='Path to the SMILES file (required if database does not exist)'
		)
		parser.add_argument('--query', required=True, type=existing_file, help='Path to the query SDF file')
		parser.add_argument('--cpu-partition', required=True, type=str, help='Partition to use for the CPU nodes')
		parser.add_argument('--gpu-partition', required=True, type=str, help='Partition to use for the GPU nodes')

		parser.add_argument('--delim', type=str, default=' ', help='Delimiter for the SMILES file (default: space)')
		parser.add_argument('--header', action='store_true', default=False, help='SMILES file has a header line')
		parser.add_argument('--pred-cpu', required=True, type=int, help='Number of cores to use for prediction')
		parser.add_argument(
			'--conf-cpu', required=True, type=int, help='Number of cores to use for conformer generation'
		)
		parser.add_argument('--search-cpu', required=True, type=int, help='Number of cores to use for shape search')
		parser.add_argument(
			'--mols-iter1', required=True, type=int, help='Number of molecules to pick for model building (iteration 1)'
		)
		parser.add_argument('--mols-iter2', required=True, type=int, help='Number of molecules to pick based on the model (iteration 2)')
		parser.add_argument('--predchunksize', required=True, type=int, help='Size of chemprop prediction chunk')
		parser.add_argument('--pred-cutoff', required=True, type=float, help='Cutoff for predictions')
		parser.add_argument('--cutoff', required=True, type=float, help='Cutoff for shape matches')

		parser.add_argument(
			'--start-from',
			type=str,
			default=FIRST_TASK,
			choices={
				'next',
				'prepare_db',
				'confgen_1',
				'align_1',
				'train',
				'predict',
				'confgen_2',
				'align_2',
				'export',
				'status',
			},
			help=(
				'Task name to start from (default: prepare_db).'
				' Works only if HASTESM has executed the previous tasks before with the same configuration.'
				" Use 'next' to start from the task after the last successfully completed task."
			),
		)

		# optional args whose default values are determined by the required args
		parser.add_argument('--search-db', type=any_path, help='Path to search DB from import-search')
		parser.add_argument('--pred-db', type=any_path, help='Path to prediction DB from import-pred')
		parser.add_argument('--matches-db', type=any_path, help='Path to shape matches DB from import-search')
		parser.add_argument(
			'--model-path', type=any_path, help='Path to trained chemprop model or directory containing it'
		)

		# confgen settings
		confgen_group = parser.add_argument_group('confgen')
		confgen_group.add_argument('-z', '--minimize', default=False, action='store_true', help='Minimize structures')
		confgen_group.add_argument(
			'-w', '--energywindow', type=float, default=25.0, help='Energy window (default: 25.0)'
		)
		confgen_group.add_argument(
			'-o', '--numconfs', type=int, default=50, help='Max. number of conformers (default: 50)'
		)
		confgen_group.add_argument('-b', '--maxrotbond', type=int, default=10, help='MAX_PER_ROT_BOND (default: 10)')
		confgen_group.add_argument(
			'-p',
			'--sample',
			default='rapid',
			choices={'rapid', 'thorough'},
			help='Conformational sampling method=rapid or thorough (default: rapid)',
		)
		confgen_group.add_argument(
			'-d',
			'--amide',
			default='trans',
			choices={'trans', 'vary', 'orig'},
			help=(
				'AMIDE_MODE=vary (allow the amide dihedral angle to take any value, not just cis or trans),'
				' orig (keep the original in the input) or trans (set to trans) (default: trans)'
			),
		)

		args = parser.parse_args()

		search_db: Path = args.search_db or args.output_dir / f'{args.name}_shape_search.db'
		matches_db: Path = args.matches_db or args.output_dir / f'{args.name}_shape_matches.db'
		pred_db: Path = args.pred_db or args.output_dir / f'{args.name}_predictions.db'
		# a directory where chemprop saves the model, its config
		# and the predictions for the test set (10% of the "picked compounds")
		model_path: Path = args.model_path or args.output_dir / f'{args.name}_trained_model'

		hastesm_params = HastesmParams(
			init_conda=args.init_conda,
			activate_conda=args.activate_conda,
			mae_format_version=args.mae_format_version,
			name=args.name,
			output_dir=args.output_dir.resolve(),
			local_dir=args.local_dir.resolve(),
			db=args.db.resolve(),
			input_smi=args.input_smi,
			delim=args.delim,
			header=args.header,
			query=args.query.resolve(),
			cpu_partition=args.cpu_partition,
			gpu_partition=args.gpu_partition,
			pred_cpu=args.pred_cpu,
			conf_cpu=args.conf_cpu,
			search_cpu=args.search_cpu,
			mols_iter1=args.mols_iter1,
			mols_iter2=args.mols_iter2,
			predchunksize=args.predchunksize,
			pred_cutoff=args.pred_cutoff,
			cutoff=args.cutoff,
			search_db=search_db.resolve(),
			matches_db=matches_db.resolve(),
			pred_db=pred_db.resolve(),
			model_path=model_path.resolve(),
			debug=args.debug,
		)

		confgen_params = ConfgenParams(
			minimize=args.minimize,
			energywindow=args.energywindow,
			numconfs=args.numconfs,
			maxrotbond=args.maxrotbond,
			sample=args.sample,
			amide=args.amide,
		)

		# Debug:
		if args.debug:
			for name, val in hastesm_params._asdict().items():
				if name == 'delim':
					print(name, '=', f'"{val}"')
				else:
					print(name, '=', val)
			for name, val in confgen_params._asdict().items():
				print(name, '=', val)

		return hastesm_params, confgen_params, args.start_from


def cleanup(local_dir: Path, task_queue: TaskQueue, logger: logging.Logger):
	"""Cancel all jobs in the queue and delete the local directory

	Args:
	----
		local_dir: Path to the local directory
		task_queue: TaskQueue instance
		logger: logging.Logger instance

	"""
	logger.info('Cleaning up')
	task_queue.cancel_all_jobs(logger)
	logger.debug(f'Deleting local directory {local_dir}')
	shutil.rmtree(local_dir, ignore_errors=True, onerror=None)


def get_completed_tasks(task_log: Path) -> Set[TaskName]:
	"""Get the set of completed tasks as TaskName objects

	Args:
	----
		task_log: Path to the task log file

	Returns:
	-------
		Set[TaskName]: Set of unique completed tasks

	"""
	task_set: Set[TaskName] = set()

	with open(task_log, 'r') as f:
		for line in f:
			if line.strip() in TaskName:
				task_set.add(TaskName(line.strip()))
	return task_set


def get_next_task(completed_tasks: Set[TaskName], logger: logging.Logger) -> TaskName:
	"""Check the task log for completed tasks in order and return the first task that has not been completed

	Args:
	----
		completed_tasks: Set of completed tasks
		logger: logging.Logger instance

	Returns:
	-------
		TaskName: Next task to run

	"""
	task_name = TaskName(FIRST_TASK)

	if len(completed_tasks) == NUM_TASKS:
		logger.error("--start-from was 'next' but all tasks have been completed. Exiting ...")
		sys.exit(0)

	while task_name in completed_tasks:
		next_task = next_task_name(task_name)
		if next_task is None:
			logger.error("--start-from was 'next' but all tasks have been completed. Exiting ...")
			sys.exit(0)
		task_name = next_task

	return task_name


def show_banner():
    print("")
    print(".___.__  .______  ._____________._._______.________._____.___ ")
    print(":   |  \ :      \ |    ___/\__ _:|: .____/|    ___/:         |")
    print("|   :   ||   .   ||___    \  |  :|| : _/\ |___    \|   \  /  |")
    print("|   .   ||   :   ||       /  |   ||   /  \|       /|   |\/   |")
    print("|___|   ||___|   ||__:___/   |   ||_.: __/|__:___/ |___| |   |")
    print("    |___|    |___|   :       |___|   :/      :           |___|")
    print("")
    print("HASTESM (macHine leArning booSTEd Shape Matching) version 0.9")
    print("Written by Samuli Näppi and Tuomo Kalliokoski, Orion Pharma")
    print("")
    print("Reference:")
    print("Kalliokoski T, Näppi S, Turku A. Machine learning-boosted shape")
    print("matching (HASTESM): searching enumerated giga-scale virtual")
    print("libraries with large conformer ensembles.")
    print("Poster at EuroQSAR 2024, Barcelona, Spain\n")


def main():
	show_banner()
	params, confgen_params, start_from = parse_arguments()

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging.DEBUG if params.debug else logging.INFO)

	hastesm_log = params.output_dir.joinpath(f'{params.name}.log')

	logging.basicConfig(
		level=logging.DEBUG,
		handlers=[logging.FileHandler(hastesm_log, encoding='utf-8', mode='w'), stream_handler],
		format='%(levelname)s (%(asctime)s): %(message)s',
		datefmt='%H:%M:%S',
	)

	logger = logging.getLogger()
	logger.info(f'Starting HASTESM workflow for {params.name}')

	task_log = params.output_dir.joinpath(f'{params.name}_task_log.txt')
	task_log_exists = task_log.is_file()

	if task_log_exists:
		completed_tasks = get_completed_tasks(task_log)
		if len(completed_tasks) == 0 or start_from == FIRST_TASK:  # task file is empty, start from beginning
			logger.info(f'Task log file {task_log} exists and is empty.')
			logger.info('Starting from the first task')

			start_from = TaskName(FIRST_TASK)
		elif start_from == 'next':
			start_from = get_next_task(completed_tasks, logger)
		else:
			# log exists, is nonempty, and start_from is a specific task name (not FIRST_TASK or "next")
			dependency = get_name_of_task_dependency(start_from)
			if dependency not in completed_tasks:
				msg = f'The dependency ({dependency}) of task {start_from} was not found in task log.'
				logger.critical(msg)
				sys.exit(1)

	elif start_from != FIRST_TASK:  # task log does not exist, start from the beginning
		msg = f'Task log file {task_log} does not exist but start_from is not {FIRST_TASK} (the default value).'
		logger.critical(msg)
		sys.exit(1)
	else:
		start_from = TaskName(FIRST_TASK)

	logger.info(f'Starting from task {start_from}.')

	task_queue = TaskQueue(logger)

	atexit.register(cleanup, params.local_dir, task_queue, logger)

	task_queue.start_from(start_from)

	while task_queue.has_next():
		completed_task = task_queue.run_next_task(params, confgen_params)
		if completed_task is not None:
			with open(task_log, 'a') as f:
				f.write(f'{completed_task.name}\n')

	logger.info('All tasks completed.')


if __name__ == '__main__':
	main()
