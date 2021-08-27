import sys
import os.path
import math
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import LabelEncoder

from bea_solver import *
from trivial_pos import *

# CONSTANTS

POS_TAGS = ['***', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'START', 'END']
UD_PETROV_TAGS = {'AUX':'VERB', 'CCONJ':'CONJ', 'INTJ':'X', 'PROPN':'NOUN', 'SCONJ':'CONJ', 'SYM':'X'}
VOTE_FOR_NULL = False
EPOCHS = 11

# Simplify a given token by replacing the digits/numbers with zeros.
def simplify_token(token):
    chars = []
    for char in token:
        if char.isdigit() or is_number(char) or (is_number(char) and is_number(token)):
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)

# Input parameters

target_language = sys.argv[1]
source_languages = sys.argv[2]
use_bea = True if sys.argv[3] == 'T' else False
use_inference_weights = True if sys.argv[4] == 'T' else False
use_weights_with_bea = True if sys.argv[5] == 'T' else False
use_weights_with_argmax = True if sys.argv[6] == 'T' else False
scale_weights = True if sys.argv[7] == 'T' else False
pos_path = sys.argv[8]
prob_path = sys.argv[9]
training_path = sys.argv[10]
forward_alignment_path = sys.argv[11]
backward_alignment_path = sys.argv[12]
gold_sets = sys.argv[13]
gold_file = sys.argv[14]
pos_output_path = sys.argv[15]
eval_output_path = sys.argv[16]

ENCODED_POS_TAGS = LabelEncoder().fit(POS_TAGS)

all_words = set()

# Reading a tabular file of sentences associated with their IDs (e.g., 'AA   This is a sentence')
# and a tabular file of probabilities associated with their IDs (e.g., 'AA   0.9 0.5 0.4 0.6')
def read_file_into_map(file_path, weights):
    pos_map = {}
    weight_map = {}
    counts = defaultdict(int)
    index = 0
    with open(file_path, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            line = line.strip()
            if len(line) == 0:
                continue;
            id = ' '.join([wp.rsplit('_', 1)[0] for wp in line.split()])
            counts[id] += 1
            key = str(counts[id])+' '+id
            pos_map[key] = line
            weight_map[key] = weights[index]
            index += 1
    return pos_map, weight_map

# Reading gold annotations
def read_gold_into_map(file_path):
    pos_map = {}
    counts = defaultdict(int)
    index = 0
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='UTF-8') as fin:
            for line in fin.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue;
                id = ' '.join([wp.rsplit('_', 1)[0] for wp in line.split()])
                poss = [wp.rsplit('_', 1)[1] for wp in line.split()]
                counts[id] += 1
                key = str(counts[id])+' '+id
                pos_map[key] = poss
                index += 1
    return pos_map

# Reading training data
def read_training_into_set(file_path, all_words):
    with open(file_path, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            words = [wp.rsplit('_', 1)[0] for wp in line.split()]
            for word in words:
                word = simplify_token(word)
                all_words.add(word)

# Reading a file of weights
def read_file_into_list(file_path):
    weights = []
    with open(file_path, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            line = line.strip()
            if len(line) == 0:
                continue;
            weights.append(line)
    return weights

# Averaging alignment probabilities over an aligned parallel corpus
def get_average_weight(file_path):
    weight = 0
    total = 0
    with open(file_path, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            columns = line.split()
            score = float(columns[0].split(':')[2])
            length1 = int(columns[0].split(':')[0])
            length2 = int(columns[0].split(':')[1])
            total += 1
            weight += score / (length1 + length2)
    return weight/total

# Reading alignment weights in the two directions
def get_alignment_weights(source_languages):
    initial_weights = []
    weights = []
    for source_language in source_languages:
        forward_prob_file = forward_alignment_path.replace('#SOURCE_LANG#', source_language)
        backward_prob_file = backward_alignment_path.replace('#SOURCE_LANG#', source_language)
        forward_weight = get_average_weight(forward_prob_file);
        reverse_weight = get_average_weight(backward_prob_file);
        weight = (forward_weight+reverse_weight)/2
        initial_weights.append(weight)
    den = 0
    for initial_weight in initial_weights:
        den += math.exp(initial_weight)
    for initial_weight in initial_weights:
        weights.append(math.exp(initial_weight)/den)
    return weights

info_lines = []


source_languages = source_languages.split(',')

# Loop over the gold sets and decoding outputs, merge the tags and evaluate.
gold_sets = gold_sets.split(',')
for gold_set in gold_sets:
    all_gold_tags = read_gold_into_map(gold_file.replace('#SET#', gold_set))
    if len(all_gold_tags) == 0:
        continue

    alignments_weights = []
    if not use_inference_weights:
        alignments_weights = get_alignment_weights(source_languages)

    all_data = []
    num_items = 0
    key_lengths = {}
    key_words = {}

    # Loop over the models and read the decoded outputs.
    for i in range(len(source_languages)):

        # Read training data.
        training_file = training_path.replace('#SOURCE_LANG#', source_languages[i])
        read_training_into_set(training_file, all_words)

        # Read the tags and their corresponding alignment probabilities.
        all_data.append({})
        pos_file = pos_path.replace("#SOURCE_LANG#", source_languages[i])
        prob_file = prob_path.replace("#SOURCE_LANG#", source_languages[i])

        # Construct the weight matrix.
        weights = []
        if use_inference_weights:
            weights = read_file_into_list(prob_file);
        else:
            weight = alignments_weights[i]
            for r in range(2000):
                weight_str = ' '
                for c in range(250):
                    weight_str += str(weight)+' '
                weights.append(weight_str.strip())
        current_pos_data, current_weight_data = read_file_into_map(pos_file, weights)
        word_index = 0
        for key in current_pos_data:
            word_pos = current_pos_data[key].split()
            words = [entry.rsplit('_', 1)[0] for entry in word_pos]
            if key not in key_lengths:
                key_lengths[key] = len(word_pos)
                key_words[key] = words
                num_items += len(word_pos)
            pos_tags = [entry.rsplit('_', 1)[1] for entry in word_pos]
            classes = [POS_TAGS.index(pos_tag) for pos_tag in pos_tags]
            weights = current_weight_data[key].split()
            weights = [float(w) for w in weights]
            all_data[i][key] = (words, classes, weights)

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
        beta_kl = np.eye(len(POS_TAGS)) * (a_v - b_v) + b_v
        zi_k, iterations = bea_infer(values, ([] if not use_weights_with_bea else (scaled_weights if scale_weights else weights)), len(POS_TAGS), beta_kl=beta_kl, prior=True)
    else:
        zi_k, iterations = mv_infer(values, ([] if not use_weights_with_bea else (scaled_weights if scale_weights else weights)), len(POS_TAGS))

    tag_assignment = ENCODED_POS_TAGS.inverse_transform((zi_k).argmax(axis=-1))

    total = 0
    correct = 0
    mismatches = 0
    match = 0
    match_petrov = 0

    IV_match = 0
    IV_total = 0

    OOV_match = 0
    OOV_total = 0

    pos_gold_map = {}
    pos_predicted_map = {}
    pos_match_map = {}

    # Read the inference output and write the multi-source tags.
    with open(pos_output_path, 'w', encoding='UTF-8') as pos_fout:
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

            gold_tags = all_gold_tags[key]

            # Calculate the predicted tags and evaluation statistics.
            for t in range(len(gold_tags)):
                total += 1
                gold_tag = gold_tags[t]
                pos_gold_map[gold_tag] = pos_gold_map.get(gold_tag, 0) + 1
                predicted_tag = tags[t]
                pos_predicted_map[predicted_tag] = pos_predicted_map.get(predicted_tag, 0) + 1
                IV = simplify_token(words[t]) in all_words
                if IV:
                    IV_total += 1
                else:
                    OOV_total += 1
                if gold_tag == predicted_tag:
                    match += 1
                    pos_match_map[predicted_tag] = pos_match_map.get(predicted_tag, 0) + 1
                    if IV:
                        IV_match += 1
                    else:
                        OOV_match += 1

                gold_petrov_tag = UD_PETROV_TAGS[gold_tag] if gold_tag in UD_PETROV_TAGS else gold_tag
                predicted_petrov_tag = UD_PETROV_TAGS[predicted_tag] if predicted_tag in UD_PETROV_TAGS else predicted_tag
                if gold_petrov_tag == predicted_petrov_tag:
                    match_petrov += 1

            pos_fout.write(' '.join([word + '_' + tag for (word, tag) in zip(words, tags)]) + '\n')

            current_row += key_lengths[key]

        accuracy_petrov = 0 if total == 0 else (match_petrov / total)
        accuracy = 0 if total == 0 else (match / total)

        # Generate tagging info.
        tag_info = ''
        for tag in POS_TAGS:
            tag_info += tag
            tag_info += ":" + (str(pos_gold_map[tag]) if tag in pos_gold_map else '0')
            tag_info += ":" + (str(pos_predicted_map[tag]) if tag in pos_predicted_map else '0')
            tag_info += ":" + (str(pos_match_map[tag]) if tag in pos_match_map else '0')
            tag_info += '-'
        tag_info = tag_info[:-1]

        info = '### ' + target_language + '-' + gold_set + '-' + gold_set + '-' + str(EPOCHS) + ' ' + tag_info + ' ' + str(match) + ':' + str(match_petrov) + '/' + str(total) + '-' + str(IV_match) + '/' + str(IV_total) + '-' + str(OOV_match) + '/' + str(OOV_total) + ' test-acc-petrov: ' + str(accuracy_petrov) + ' test-acc: ' + str(accuracy)
        info_lines.append(info)

# Write the results.
with open(eval_output_path, 'w', encoding='UTF-8') as eval_fout:
    for info_line in info_lines:
        eval_fout.write(info_line+'\n')
