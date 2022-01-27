from collections import defaultdict
import json
import itertools as it
from os import error
from pathlib import Path
from tqdm import tqdm
from typing import List, NamedTuple, Optional
from pprint import pprint
from collections import defaultdict

class EventFrame(NamedTuple):
	args: List[str]
	trigger: Optional[str]
	start: int
	end: int
	type: str
	passage_id: str
	sentence_id: str

def make_file_contents(pmcid:str, base_dir:Path = Path()) -> List[str]:
	events_path = base_dir / f"{pmcid}.uaz.events.json"
	sentences_path = base_dir / f"{pmcid}.uaz.sentences.json"
	entities_path  = base_dir / f"{pmcid}.uaz.entities.json"

	# Build the sentence frames cache
	with sentences_path.open() as f:
		sentences = {frame['frame-id']:frame for frame in json.load(f)['frames']}

	# Entity frames cache
	with entities_path.open() as f:
		entities = {}
		for frame in json.load(f)['frames']:
			# Ignore other frame types (such as context)
			if frame['frame-type'] == 'entity-mention':
				start, end = int(frame['start-pos']['offset']), int(frame['end-pos']['offset'])
				entity_id = frame['frame-id']
				entities[entity_id] = (start, end)

	# Event mentions
	with events_path.open() as f:
		simple_events, complex_events = {}, {}
		successes, errors = 0, 0
		for frame in json.load(f)['frames']:
			try:
				# Get the data from this frmae
				trigger = frame['trigger'] if 'trigger' in frame else None
				start, end = int(frame['start-pos']['offset']), int(frame['end-pos']['offset'])
				passage_id = frame['start-pos']['reference']
				sentence_id = frame['sentence']
				event_type = frame['subtype'] if 'subtype' in frame else frame['type']
				arguments = [a['arg'] for a in frame['arguments']]

				ef = EventFrame(arguments, trigger, start, end, event_type, passage_id, sentence_id)

				if 'trigger' not in frame:
					complex_events[frame['frame-id']] = ef
				else:
					simple_events[frame['frame-id']] = ef
				
				successes += 1
			except Exception as ex:
				errors += 1

	all_events = list(sorted(list(simple_events.values())+list(complex_events.values()), key=lambda ef:ef.sentence_id))
	events_by_sent = it.groupby(all_events, key= lambda ef: ef.sentence_id)

	blocks = list()

	for sent_id, events in events_by_sent:
		sentence_frame = sentences[sent_id]
		passage = sentences[sentence_frame['passage']]['text']
		sent_start = int(sentence_frame['start-pos']['offset'])
		sentence_text = passage[sentence_frame['start-pos']['offset']:sentence_frame['end-pos']['offset']]
		tokens = sentence_text.split()

		labels = set()
		tags = defaultdict(set)

		for ef in sorted(events, key= lambda ef: (ef.start, ef.end)):
			labels.add(ef.type)
			args = list(sorted((entities[a] for a in ef.args if a in entities), key=lambda a: a[0]))
			index = 0
			for token_ix, token in enumerate(tokens):
				if token == ef.trigger:
					tags[token_ix].add('TRI')
				elif len(args) == 0:
					tags[token_ix].add('O')
				else:
					arg = args[0]
					start, end = (arg[0] - sent_start), (arg[1] - sent_start)
					if start <= index <= end:
						tags[token_ix].add('ARG')
						args = args[1:]
					else:
						tags[token_ix].add('O')

				index += len(token) + 1 # Add one to account for the whitespace

		# Fix the labels
		fixed_tags = list()
		for l in tags.values():
			if 'O' in l and len(l) == 1:
				fixed_tags.append('O')
			else:
				if 'O' in l:
					l.remove('O')
				fixed_tags.append(next(iter(l)))

		#Build the block
		blocks.append(' '.join(labels))
		blocks.append('\n')
		blocks.append('\t'.join(fixed_tags))
		blocks.append('\n')
		blocks.append('\t'.join(tokens))
		blocks.append('\n')
		blocks.append('\n')
			
	return blocks, successes, errors

def process_dir(input_directory:Path, output_directory:Optional[Path] = Path()):
	# Get the pmcids in the directory
	pmcids = {p.name.split('.')[0] for p in input_directory.glob('**/*.events.json')}
	total_successes, total_errors = 0, 0
	for pmcid in tqdm(pmcids, desc='Parsing files'):
		blocks, successes, errors = make_file_contents(pmcid, input_directory)
		total_errors += errors
		total_successes += successes


		output_file = output_directory / f'{pmcid}.txt'
		with output_file.open('w') as f:
			f.writelines(blocks)


if __name__ == "__main__":
	typer.run(process_dir)






