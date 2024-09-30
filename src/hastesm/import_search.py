"""Must be run with $SCHRODINGER/run, cannot be imported directly"""

import argparse
import sqlite3
import sys
from pathlib import Path


def import_search(args):
	"""Import shape searching results.

	Args:
	----
		args: Parsed arguments

	"""
	try:
		from schrodinger import structure
	except ImportError:
		print('Schrödinger Suite Python environment not detected, did you start HASTESM with $SCHRODINGER/run?')
		sys.exit(1)

	filenames = [str(p) for p in args.output_dir.glob(f'{args.name}_iter{args.iteration}_cpu*_align.maegz')]

	if not filenames:
		print('No shape matching results found')
		sys.exit(1)
	if not args.db.is_file():
		print(f'HASTESM database {args.db} not found.')
		sys.exit(1)
	if args.iteration != 1:
		if not args.search_db.is_file():
			print(f'Iteration is not 1 but shape searching DB ({args.search_db}) is missing')
			sys.exit(1)
		if not args.matches_db.is_file():
			print(f'Iteration is not 1 but matches DB ({args.matches_db}) is missing.')
			sys.exit(1)

	sims = {}
	structs = {}
	print(f'Importing shape matching results to {args.search_db} and matches to {args.matches_db}')

	reader = structure.MultiFileStructureReader(filenames, reader_class=structure.MaestroReader)

	conn = sqlite3.connect(args.db)
	c = conn.cursor()

	for st in reader:
		hastenid = int(st.property['s_m_title'])
		shape_similarity = float(st.property['r_phase_Shape_Sim'])

		if hastenid not in sims or sims[hastenid] < shape_similarity:
			sims[hastenid] = shape_similarity
			if shape_similarity >= args.cutoff:
				structs[hastenid] = st
				# smilesid = c.execute(f'SELECT smilesid FROM data WHERE hastenid = {hastenid}').fetchone()[0]
				# st.property['s_m_title'] = smilesid
				# structs[hastenid] = structure.write_ct_to_string(st)

	print(f'Imported {len(sims)} molecules')

	# get smiles IDs of all matches in one JOIN query 
	c.execute("CREATE TEMPORARY TABLE temp_hastenids (hastenid INTEGER PRIMARY KEY)")
	c.executemany("INSERT INTO temp_hastenids (hastenid) VALUES (?)", 
		((hastenid,) for hastenid in structs.keys())
	)

	query = """
	SELECT t.hastenid, d.smilesid
	FROM temp_hastenids t
	JOIN data d ON t.hastenid = d.hastenid
	"""
	c.execute(query)
	hastenid_to_smilesid = dict(c.fetchall())

	# Clean up temporary table
	c.execute("DROP TABLE temp_hastenids")

	# write structures to strings with the correct IDs as titles
	for hastenid, st in structs.items():
		smilesid = hastenid_to_smilesid.get(hastenid)
		if smilesid:
			st.property['s_m_title'] = smilesid
			structs[hastenid] = structure.write_ct_to_string(st)

	to_search_db = []
	for hastenid in sims:
		to_search_db.append((hastenid, sims[hastenid], args.iteration))

	to_shape_db = []
	for hastenid in structs:
		to_shape_db.append((hastenid, sims[hastenid], args.iteration, structs[hastenid]))

	search_conn = sqlite3.connect(args.search_db)
	c = search_conn.cursor()
	c.execute(
		"""
		CREATE TABLE IF NOT EXISTS searching_data
			(hastenid INTEGER PRIMARY KEY, similarity NUMERIC, search_iteration INTEGER)
		"""
	)
	c.executemany(
		'INSERT OR REPLACE INTO searching_data (hastenid,similarity,search_iteration) VALUES (?,?,?)', to_search_db
	)
	search_conn.commit()
	search_conn.close()

	matches_conn = sqlite3.connect(args.matches_db)
	c = matches_conn.cursor()
	c.execute(
		"""
		CREATE TABLE IF NOT EXISTS searching_data
			(hastenid INTEGER PRIMARY KEY, similarity NUMERIC, search_iteration INTEGER, structure TEXT)
		"""
	)
	c.executemany(
		"""
		INSERT OR REPLACE INTO searching_data
			(hastenid,similarity,search_iteration,structure) VALUES (?,?,?,?)
		""",
		to_shape_db,
	)
	matches_conn.commit()
	matches_conn.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', required=True, help='Unique name for the search')
	parser.add_argument('--db', required=True, type=Path, help='Path to the HASTESM database')
	parser.add_argument('--output-dir', required=True, type=Path, help='Path to the output directory on the NFS drive')
	parser.add_argument('--search-db', required=True, type=Path, help='Path to the shape searching database')
	parser.add_argument('--matches-db', required=True, type=Path, help='Path to the shape matching database')
	parser.add_argument('--cutoff', required=True, type=float, help='Cutoff for shape matching')
	parser.add_argument('--iteration', required=True, type=int, help='Shape matching iteration number')
	parser.add_argument('--debug', action='store_true', help='Enable debug logging')
	args = parser.parse_args()

	import_search(args)
