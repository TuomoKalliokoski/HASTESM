import csv
import gzip
import sqlite3
from logging import Logger
from pathlib import Path
from typing import List

from configargparse import ArgumentError

from ..hastesm_types import HastesmParams


def import_smiles(db: Path, input_smi: Path, delim: str, header: bool, logger: Logger) -> None:
	"""Import SMILES file to HASTEN database

	Args:
	----
		db: Path to the HASTESM SMILES database
		input_smi: Path to the SMILES file (CSV format, first column must be the SMILES string)
		delim: Delimiter for the SMILES file
		header: Whether the SMILES file has a header
		logger: logging.Logger instance

	Raises:
	------
		ValueError: If SMILES file has less than 2 columns

	"""
	logger.info(f"Importing SMILES to {'a new' if not db.exists() else 'an existing'} database {db}")

	conn = sqlite3.connect(db)
	c = conn.cursor()
	c.execute('CREATE TABLE IF NOT EXISTS data (hastenid INTEGER PRIMARY KEY, smiles TEXT, smilesid TEXT)')

	opener = gzip.open if input_smi.suffix == '.gz' else open

	with opener(input_smi, 'rt') as smilesfile:
		reader = csv.reader(smilesfile, delimiter=delim)
		first_row = next(reader)
		if len(first_row) < 2:
			raise ValueError('The input SMILES file should have at least two columns.')

		if not header:
			smilesfile.seek(0)

		c.executemany('INSERT INTO data(smiles, smilesid) VALUES (?, ?)', ((row[0], row[1]) for row in reader))

		logger.info(f'Imported {c.rowcount} molecules')

		smilesfile.close()
		conn.commit()
		conn.close()


def prepare_db(params: HastesmParams, logger: Logger, job_ids: List[str]) -> None:
	"""Prepares the HASTESM database

	Args:
	----
		params: Parsed HastesmParams from the main file
		logger: logging.Logger instance
		job_ids: List of slurm job ids (unused)

	Raises:
	------
		ArgumentError: If the SMILES database file does not exist and SMILES file is not provided
		FileNotFoundError: If neither the SMILES database file nor the SMILES file provided exist
		FileNotFoundError: If the provided SMILES file does not exist

	"""
	db_exists = params.db.is_file()
	input_smi_provided = params.input_smi is not None
	input_smi_exists = input_smi_provided and params.input_smi.is_file()

	if not db_exists:
		if not input_smi_provided:
			raise ArgumentError('SMILES database file not found and no input SMILES file provided.')
		elif not input_smi_exists:
			raise FileNotFoundError(
				f'SMILES database file not found and provided input SMILES file {params.input_smi} does not exist.'
			)
		else:
			logger.info(f'SMILES database file not found. Importing SMILES from {params.input_smi}')
			import_smiles(params.db, params.input_smi, params.delim, params.header, logger)
	else:
		if input_smi_provided and not input_smi_exists:
			raise FileNotFoundError(f'Error: Provided input SMILES file {params.input_smi} does not exist.')
		elif input_smi_exists:
			logger.info('Database file exists')
			conn = sqlite3.connect(params.db)
			num_rows = conn.execute('SELECT COUNT(*) FROM data').fetchone()[0]
			conn.close()
			logger.info(f'Number of rows in the database: {num_rows}')
			print(f'Do you want to import additional SMILES from {params.input_smi}? (y/n)')
			if input() == 'y':
				logger.info(f'Importing SMILES from {params.input_smi}')
				import_smiles(params.db, params.input_smi, params.delim, params.header, logger)
			else:
				logger.info(f'Using existing database: {params.db}')
		else:
			logger.info(f'Using existing database: {params.db}')
