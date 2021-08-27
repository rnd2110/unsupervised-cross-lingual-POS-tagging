import sys
from collections import defaultdict

from trivial_pos import *

# CONSTANTS

POS_NUMBER = 'NUM'
POS_SYMBOL = 'SYM'
POS_PUNCTUATION = 'PUNCT'
POS_NULL = '***'
POS_ALL = 'ALL'
MIN_ALIGNMENT_PROBABILITY = 0.1
MIN_TYPE_PERCENTAGE = 0.3

# Reading alignment keys
def read_file_into_list(file):
    list = []
    with open(file, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            list.append(line.strip())
    return list

# Reading a GIZA++ vocabulary file
def read_vocabulary(vocabulary_file):
    vocabulary_map = {}
    with open(vocabulary_file, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            columns = line.strip().split()
            vocabulary_map[int(columns[0])] = columns[1]
    return vocabulary_map

# Reading a tabular file of probabilities associated with their IDs (e.g., 'AA   0.9 0.5 0.4 0.6')
def read_word_alignment_probabilities(alignment_probability_file, source_vocabulary_map, target_vocabulary_map):
    alignment_probability_map = {}
    with open(alignment_probability_file, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            columns = line.strip().split()
            source_index = int(columns[0])
            target_index = int(columns[1])
            source_word = source_vocabulary_map[source_index] if source_index in source_vocabulary_map else None
            target_word = target_vocabulary_map[target_index] if target_index in target_vocabulary_map else None
            probability = float(columns[2])
            if source_word and target_word:
                if source_word not in alignment_probability_map:
                    alignment_probability_map[source_word] = {}
                alignment_probability_map[source_word][target_word] = probability
    return alignment_probability_map

# Reading a tabular file of annotated sentences associated with their IDs (e.g., 'AA   This_PRON is_AUX a_DET sentence_NOUN')
def read_annotations(annotation_file):
    annotation_map = {}
    with open(annotation_file, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            columns = line.strip().split('\t')
            key = columns[0]
            annotation_map[key] = []
            annotations = columns[1].split()
            for annotation in annotations:
                word_pos = annotation.rsplit('_', 1)
                word, pos = word_pos[0], word_pos[1]
                annotation_map[key].append([word, pos])
    return annotation_map

# Reading a tabular file of sentences associated with their IDs (e.g., 'AA   This is a sentence')
def read_data(data_file):
    data_map = {}
    with open(data_file, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            columns = line.strip().split('\t')
            key = columns[0]
            words = columns[1].split()
            data_map[key] = words
    return data_map

# Filtering out type constraints whose POS probabilities are below some threshold + Coupling token and type constraints.
def filter_type_constraints(target_annotations):
    word_pos_counts = {}
    for key in target_annotations:
        for annotation in target_annotations[key]:
            word, pos = annotation[0], annotation[1]
            if pos == POS_NULL:
                continue
            if word not in word_pos_counts:
                word_pos_counts[word] = defaultdict(int)
            word_pos_counts[word][pos] += 1
            word_pos_counts[word][POS_ALL] += 1

    allowed_word_pos = {}
    for word in word_pos_counts:
        allowed_word_pos[word] = []
        for pos in word_pos_counts[word]:
            if pos != POS_ALL:
                percentage = word_pos_counts[word][pos]/word_pos_counts[word][POS_ALL]
                if percentage >= MIN_TYPE_PERCENTAGE:
                    allowed_word_pos[word].append(pos)

    for key in target_annotations:
        for index, annotation in enumerate(target_annotations[key]):
            word, pos = annotation[0], annotation[1]
            if word in allowed_word_pos and pos not in allowed_word_pos[word]:
                annotation[1] = POS_NULL
                annotation[2] = 0

# Input parameters
key_path = sys.argv[1]
forward_alignment_path = sys.argv[2]
forward_source_vocabulary_path = sys.argv[3]
forward_target_vocabulary_path = sys.argv[4]
forward_alignment_probability_path = sys.argv[5]
backward_alignment_path = sys.argv[6]
backward_source_vocabulary_path = sys.argv[7]
backward_target_vocabulary_path = sys.argv[8]
backward_alignment_probability_path = sys.argv[9]
tagged_source_path = sys.argv[10]
target_data_path = sys.argv[11]
pos_output_path = sys.argv[12]
prob_output_path = sys.argv[13]

# Read keys.
keys = read_file_into_list(key_path)

# Read GIZA++ forward alignments.
forward_alignments = read_file_into_list(forward_alignment_path)

# Read GIZA++ backward alignments.
backward_alignments = read_file_into_list(backward_alignment_path)

if len(forward_alignments) != len(backward_alignments):
    print("Forward and backward alignments are not equal in size!")
    exit(0)

# Keep track of bidirectional alignments (index-based).
alignment_indexes = {}
for i in range(len(forward_alignments)):
    alignment_indexes[keys[i]] = {}
    forward_alignment_indexes = forward_alignments[i].strip().split()[1:]
    backward_alignment_indexes = backward_alignments[i].strip().split()[1:]
    for alignment_index in forward_alignment_indexes:
        if alignment_index in backward_alignment_indexes:
            indexes = [int(idx) for idx in alignment_index.split('-')]
            source_index = indexes[0]
            target_index = indexes[1]
            if target_index not in alignment_indexes[keys[i]]:
                alignment_indexes[keys[i]][target_index] = []
            alignment_indexes[keys[i]][target_index].append(source_index)

# Replace the indexes in the forward alignments by their corresponding words.
forward_source_vocabulary = read_vocabulary(forward_source_vocabulary_path)
forward_target_vocabulary = read_vocabulary(forward_target_vocabulary_path)
forward_alignment_probabilities = read_word_alignment_probabilities(forward_alignment_probability_path, forward_source_vocabulary, forward_target_vocabulary)

# Replace the indexes in the backward alignments by their corresponding words.
backward_source_vocabulary = read_vocabulary(backward_source_vocabulary_path)
backward_target_vocabulary = read_vocabulary(backward_target_vocabulary_path)
backward_alignment_probabilities = read_word_alignment_probabilities(backward_alignment_probability_path, backward_target_vocabulary, backward_source_vocabulary)

# Assign the alignment probabilities to the average of the bidirectional alignments.

alignment_probabilities = {}
for source_word in forward_alignment_probabilities:
    alignment_probabilities[source_word] = {}
    for target_word in forward_alignment_probabilities[source_word]:
        forward_alignment_probability = forward_alignment_probabilities[source_word][target_word]
        backward_alignment_probability = 0
        if target_word in backward_alignment_probabilities and source_word in backward_alignment_probabilities[target_word]:
            backward_alignment_probability = backward_alignment_probabilities[target_word][source_word]
        alignment_probability = (forward_alignment_probability + backward_alignment_probability) / 2
        alignment_probabilities[source_word][target_word] = alignment_probability

for target_word in backward_alignment_probabilities:
    for source_word in backward_alignment_probabilities[target_word]:
        backward_alignment_probability = backward_alignment_probabilities[target_word][source_word]
        if source_word not in alignment_probabilities or target_word not in alignment_probabilities[source_word]:
            alignment_probability = backward_alignment_probability / 2
            if source_word not in alignment_probabilities:
                alignment_probabilities[source_word] = {}
            alignment_probabilities[source_word][target_word] = alignment_probability

# Read source annotations.
source_annotations = read_annotations(tagged_source_path)

# Read target text.
target_data = read_data(target_data_path)
target_annotations = {}

target_probabilities = {}
target_analyses = {}

# Loop over each line in the parallel data and do projection (key-based).
for key in source_annotations:
    if key not in target_data:
        continue
    target_annotations[key] = []
    target_probabilities[key] = []
    current_analysis = []
    # Read source data.
    processed_source_indexes = []
    source_words = [a[0] for a in source_annotations[key]]
    source_poses = [a[1] for a in source_annotations[key]]
    target_words = target_data[key]

    # Loop over the target words.
    for target_index, target_word in enumerate(target_words):

        projected_pos = POS_NULL
        pos_probability = 0

        # Tag the trivial cases.
        if is_number(target_word):
            projected_pos = POS_NUMBER
            pos_probability = 1
        elif is_symbol(target_word):
            projected_pos = POS_NUMBER
            pos_probability = 1
        elif is_punctuation(target_word):
            projected_pos = POS_PUNCTUATION
            pos_probability = 1

        # If not a trivial target word, find the source word that is aligned to the current target word with the highest alignment probability.
        if projected_pos == POS_NULL and key in alignment_indexes and target_index in alignment_indexes[key]:
            source_indexes = alignment_indexes[key][target_index]
            selected_source_index = -1
            max_alignment_probability = -1
            for source_index in source_indexes:
                source_word = source_words[source_index]
                source_pos = source_poses[source_index]
                if source_index not in processed_source_indexes and source_word in alignment_probabilities and target_word in alignment_probabilities[source_word] and not is_trivial(source_word) and not source_pos == POS_SYMBOL and not source_pos == POS_PUNCTUATION:
                    alignment_probability = alignment_probabilities[source_word][target_word]
                    if alignment_probability > max_alignment_probability:
                        selected_source_index = source_index
                        max_alignment_probability = alignment_probability
            if selected_source_index != -1 and max_alignment_probability >= MIN_ALIGNMENT_PROBABILITY:
                processed_source_indexes.append(selected_source_index)
                projected_pos = source_poses[selected_source_index]
                pos_probability = max_alignment_probability

        # If no alignment found, project from the source word that equals the current target word, if any.
        if projected_pos == POS_NULL:
            for source_index, source_word in enumerate(source_words):
                if source_index not in processed_source_indexes and source_word == target_word:
                    processed_source_indexes.append(source_index)
                    projected_pos = source_poses[source_index]
                    pos_probability = 1

        # Store projection information (token constraints).
        target_annotations[key].append([target_word, projected_pos, pos_probability])

# Filter out unlikely type constraints, and couple token and type constrains.
filter_type_constraints(target_annotations)

# Write out target data.
with open(pos_output_path, 'w', encoding='UTF-8') as pos_fout:
    with open(prob_output_path, 'w', encoding='UTF-8') as prob_fout:
        for key in target_annotations:
            pos_fout.write(key + '\t' + ' '.join([a[0]+'_'+a[1] for a in target_annotations[key]]) + '\n')
            prob_fout.write(key + '\t' + ' '.join([str(a[2]) for a in target_annotations[key]]) + '\n')
