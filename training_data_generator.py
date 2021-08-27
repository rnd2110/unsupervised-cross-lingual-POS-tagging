import sys

# CONSTANTS

POS_NULL = '***'
MIN_SENTENCE_WEIGHT = 0.5

# Reading a tabular file of annotated sentences associated with their IDs (e.g., 'AA   This_PRON is_AUX a_DET sentence_NOUN')
def read_annotations(pos_file):
    annotation_map = {}
    with open(pos_file, 'r', encoding='UTF-8') as fin:
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

# Reading a tabular file of annotation weights (e.g., 'AA   0.94 0.22 0.83 0.55')
def read_probabilities(prob_file):
    weight_map = {}
    with open(prob_file, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            columns = line.strip().split('\t')
            key = columns[0]
            weights = [float(w) for w in columns[1].split()]
            weight_map[key] = weights
    return weight_map

# Computing sentence weight given density and alignment confidence
def compute_sentence_weight(target_annitations, target_weights):
    alignment_confidence = 0
    tagged_token_count = 0
    for i in range(len(target_annitations)):
        pos = target_annitations[i][1]
        probability = target_weights[i]
        if pos != POS_NULL:
            tagged_token_count += 1
            alignment_confidence += probability
    alignment_confidence = alignment_confidence / tagged_token_count
    density = tagged_token_count/len(target_annitations)
    weight = (2 * alignment_confidence * density) / (alignment_confidence + density) if (alignment_confidence != 0 and density != 0) else 0
    return weight


pos_path = sys.argv[1]
prob_path = sys.argv[2]
training_path = sys.argv[3]

# Read the projected annotations
target_annotations = read_annotations(pos_path)

# Read annotation weights
target_weights = read_probabilities(prob_path)

# Loop over the sentences and weight them.
weighted_sentences = {}
for key in target_annotations:
    weight = compute_sentence_weight(target_annotations[key], target_weights[key])
    weighted_sentences[key] = weight

# Sort the weighted sentences.
weighted_sentences = {k: v for k, v in sorted(weighted_sentences.items(), key=lambda item: item[1], reverse=True)}

# Write out training data.
# Only consider the top weighted sentences (whose probabilities >= MIN_SENTENCE_WEIGHT) as training instances.
with open(training_path, 'w', encoding='UTF-8') as fout:
    for key in weighted_sentences:
        weight = weighted_sentences[key]
        if weight < MIN_SENTENCE_WEIGHT:
            break
        fout.write(' '.join([a[0]+'_'+a[1] for a in target_annotations[key]])+'\n')
