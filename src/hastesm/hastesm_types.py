"""Type definitions for HASTESM"""

from enum import Enum, EnumMeta
from pathlib import Path
from typing import List, Literal, NamedTuple, Optional


class HastesmParams(NamedTuple):

	"""Parsed arguments for HASTESM"""

	init_conda: str
	activate_conda: str
	mae_format_version: str
	name: str
	output_dir: Path
	local_dir: Path
	db: Path
	input_smi: Optional[Path]
	delim: str
	header: bool
	query: Path
	cpu_partition: str
	gpu_partition: str
	pred_cpu: int
	conf_cpu: int
	search_cpu: int
	mols_iter1: int
	mols_iter2: int
	predchunksize: int
	pred_cutoff: float
	cutoff: float
	search_db: Path
	pred_db: Path
	matches_db: Path
	model_path: Path
	debug: bool


class ConfgenParams(NamedTuple):

	"""Parsed arguments for conformer generation"""

	minimize: bool
	energywindow: float
	numconfs: int
	maxrotbond: int
	sample: Literal['rapid', 'thorough']
	amide: Literal['trans', 'vary', 'orig']


class MetaEnumContains(EnumMeta):
	def __contains__(self, value) -> bool:
		try:
			self(value)
		except ValueError:
			return False
		return True


class EnumContains(Enum, metaclass=MetaEnumContains):
	pass


class TaskName(str, EnumContains):
	PREPARE_DB = 'prepare_db'
	CONFGEN_1 = 'confgen_1'
	ALIGN_1 = 'align_1'
	TRAIN = 'train'
	PREDICT = 'predict'
	CONFGEN_2 = 'confgen_2'
	ALIGN_2 = 'align_2'
	EXPORT = 'export'
	STATUS = 'status'

	def __repr__(self) -> str:
		return self.value

	def __str__(self) -> str:
		return self.value
