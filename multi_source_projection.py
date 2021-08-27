import sys
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder

from bea_solver import *

# CONSTANTS
POS_TAGS = ['***', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
VOTE_FOR_NULL = False

# Input parameters
target_language = sys.argv[1]
source_languages = sys.argv[2]
use_bea= True if sys.argv[3] == 'T' else False
use_weights_with_bea = True if sys.argv[4] == 'T' else False
use_weights_with_argmax = True if sys.argv[5] == 'T' else False
scale_weights = True if sys.argv[6] == 'T' else False
pos_path = sys.argv[7]
prob_path = sys.argv[8]
pos_output_path = sys.argv[9]
prob_output_path = sys.argv[10]

ENCODED_POS_TAGS = LabelEncoder().fit(POS_TAGS)

# Reading a tabular file of sentences associated with their IDs (e.g., 'AA   This is a sentence')
def read_file(file_path):
    map = {}
    with open(file_path, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            columns = line.split('\t')
            id = columns[0]
            text = columns[1]
            map[id] = text
    return map

all_data = []
num_items = 0
key_lengths = {}
key_words = {}

# Loop over the source languages and read the projections.
source_languages = source_languages.split(',')
for i in range(len(source_languages)):
    all_data.append({})
    pos_file = pos_path.replace("#SOURCE_LANG#", source_languages[i])
    prob_file = prob_path.replace("#SOURCE_LANG#", source_languages[i])
    current_pos_data = read_file(pos_file)
    current_weight_data = read_file(prob_file)
    word_index = 0
    for key in current_pos_data:
        word_pos = current_pos_data[key].split()
        words = [entry.rsplit('_', 1)[0] for entry in word_pos]
        if key not in key_lengths:
            key_lengths[key] = len(word_pos)
            key_words[key] = words
            num_items += len(word_pos)

        # Read the tags and their corresponding alignment probabilities.
        pos_tags = [entry.rsplit('_', 1)[1] for entry in word_pos]
        classes = [POS_TAGS.index(pos_tag) for pos_tag in pos_tags]
        weights = current_weight_data[key].split()
        weights = [float(w) for w in weights]
        all_data[i][key] = (words, classes, weights)

# Construct the weight matrix.
values = np.empty(shape=(num_items, len(source_languages)), dtype=int)
weights = np.empty(shape=(num_items, len(source_languages)), dtype=float)
scaled_weights = np.empty(shape=(num_items, len(source_languages)), dtype=int)

current_row = 0
for key in key_lengths:
    for i in range(len(all_data)):
        if key in all_data[i]:
            current_classes = all_data[i][key][1]
            current_weights = all_data[i][key][2]
            for j in range(key_lengths[key]):
                current_index = current_row + j
                values[current_index, i] = current_classes[j]
                weights[current_index, i] = current_weights[j]
                scaled_weights[current_index, i] = math.floor(current_weights[j]*10)
    current_row += key_lengths[key]

# Do inference.
unknowns = np.where(~values.any(axis=1))[0]
zi_k = None
iterations = 1
if use_bea:
    a_v, b_v = 1, 1
    beta_kl = np.eye(len(POS_TAGS)) * (a_v-b_v) + b_v
    zi_k, iterations = bea_infer(values, ([] if not use_weights_with_bea else (scaled_weights if scale_weights else weights)), len(POS_TAGS), beta_kl=beta_kl, prior=True)
else:
    zi_k, iterations = mv_infer(values, (scaled_weights if scale_weights else weights), len(POS_TAGS))

tag_assignment = ENCODED_POS_TAGS.inverse_transform((zi_k).argmax(axis=-1))

mismatches = 0

# Read the inference output and write the multi-source tags.
with open(pos_output_path, 'w', encoding='UTF-8') as pos_fout:
    with open(prob_output_path, 'w', encoding='UTF-8') as prob_fout:
        current_row = 0
        for key in key_lengths:
            words = key_words[key]
            tags = []
            final_weights = []
            for j in range(key_lengths[key]):
                current_index = current_row + j
                output = [0 if i not in values[current_index] else zi_k[current_index][i] for i in range(len(POS_TAGS))]
                output = np.array(output)

                if use_weights_with_argmax:
                    aggregated_weights = np.zeros(len(POS_TAGS))
                    for idx, v in enumerate(values[current_index]):
                        aggregated_weights[v] += (scaled_weights[current_index] if scale_weights else weights[current_index])[idx]
                    output = np.multiply(output, aggregated_weights)

                tag_id = 0
                if not np.any(output[1:]):
                    tag_id = 0
                else:
                    tag_id = (output if VOTE_FOR_NULL else output[1:]).argmax(axis=-1)+(0 if VOTE_FOR_NULL else 1)

                selected_tag = POS_TAGS[tag_id]
                tags.append(selected_tag)
                if selected_tag != tag_assignment[current_index]:
                    mismatches += 1

                aggregated_weight = 0
                for idx, v in enumerate(values[current_index]):
                    if v == tag_id:
                        aggregated_weight += weights[current_index][idx]
                final_weights.append(aggregated_weight)

            pos_fout.write(key+'\t'+' '.join([word + '_' + tag for (word, tag) in zip(words, tags)]) + '\n')
            prob_fout.write(key + '\t' + ' '.join([str(w) for w in final_weights]) + '\n')

            current_row += key_lengths[key]

print(target_language, "Iterations:", iterations, "Mismatches:", mismatches)