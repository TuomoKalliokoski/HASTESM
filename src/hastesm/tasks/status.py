import sqlite3
from logging import Logger
from pathlib import Path
from typing import List

from ..hastesm_types import HastesmParams


def print_table(header, rows, column_widths):
	"""Helper function to print a table with aligned columns."""
	header_row = '| ' + ' | '.join(f'{col:^{width}}' for col, width in zip(header, column_widths)) + ' |'
	separator_row = '| ' + ' | '.join('-' * width for width in column_widths) + ' |'

	print(header_row)
	print(separator_row)

	for row in rows:
		print('| ' + ' | '.join(f'{str(col):^{width}}' for col, width in zip(row, column_widths)) + ' |')


def show_status(name: str, db: Path, search_db: Path) -> None:
	"""Show status of the search

	Args:
	----
		name: Name of the job
		db: Path to the HASTESM SMILES database
		search_db: Path to the search results database

	"""
	print('Jobname                                 :', name)
	print('\nDatabase                                :', db)
	print('Searching results database              :', search_db)

	with sqlite3.connect(db) as conn:
		c = conn.cursor()
		number_of_mols = c.execute('SELECT MAX(_ROWID_) FROM data LIMIT 1').fetchone()[0]

	with sqlite3.connect(search_db) as conn:
		c = conn.cursor()
		best_similarity = c.execute('SELECT MAX(similarity) FROM searching_data LIMIT 1').fetchone()[0]
		number_of_searched = c.execute(
			'SELECT COUNT(*) FROM searching_data WHERE similarity NOT NULL LIMIT 1'
		).fetchone()[0]

	print('Total number of molecules to search     :', number_of_mols)
	print(
		'Number of molecules through shape search:',
		number_of_searched,
		f'({round(number_of_searched/number_of_mols*100.0, 3)}%)',
	)
	print('Highest similarity observed             :', round(best_similarity, 3))

	print('\nMatched compounds at different levels of shape similarity:\n')
	cut_offs = [0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.75]
	header = ['Iter'] + [str(cut_off) for cut_off in cut_offs]

	summary_table = []
	for i in [1, 2]:
		row = [i]
		for cur_cutoff in cut_offs:
			hits = c.execute(
				'SELECT COUNT(*) FROM searching_data WHERE search_iteration <= ? AND similarity >= ?', (i, cur_cutoff)
			).fetchone()[0]
			row.append(hits)
		summary_table.append(row)

	column_widths = [4] + [7] * len(cut_offs)
	print_table(header, summary_table, column_widths)


def status(params: HastesmParams, logger: Logger, job_ids: List[str]) -> None:
	"""Shows the status of the HASTESM search

	Args:
	----
		params: Parsed HastesmParams from the main file
		logger: logging.Logger instance
		job_ids: List of slurm job ids (unused)

	Raises:
	------
		FileNotFoundError: If the search database does not exist

	"""
	if not params.search_db.is_file():
		msg = f'No searching results found for this job ({params.search_db})'
		logger.critical(msg)
		raise FileNotFoundError(msg)

	logger.debug(f'status: {params.name} {params.db} {params.search_db}')
	show_status(params.name, params.db, params.search_db)
