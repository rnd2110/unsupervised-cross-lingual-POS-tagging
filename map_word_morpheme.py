import sys

# Reading a tabular file of sentences associated with their IDs (e.g., 'AA   This is a sentence')
def read_file(file_path):
    map = {}
    with open(file_path, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            line = line.strip()
            columns = line.split('\t')
            id = columns[0]
            text = columns[1]
            map[id] = text
    return map

# Given morphemes and their POS, assign the corresponding word the POS of the stem morpheme.
def get_pos_by_stem(words, morphs, poss, stemmorphs, probs):
    word_index = 0
    output_poss = []
    output_probs = []
    current_word = ''
    previous_morphs = ''
    stem_pos = None
    stem_prob = None
    for i in range(len(morphs)):
        current_word += morphs[i]
        stemmorpheme_splits = stemmorphs[word_index].split('+')
        if stemmorpheme_splits[0] == previous_morphs and stemmorpheme_splits[1] == morphs[i]:
            stem_pos = poss[i]
            stem_prob = probs[i]
        previous_morphs += morphs[i]
        if current_word == words[word_index]:
            output_poss.append(stem_pos)
            output_probs.append(stem_prob)
            word_index += 1
            current_word = ''
            previous_morphs = ''
            stem_pos = None
            stem_prob = None
    if len(words) != len(output_poss):
        print(len(words), len(output_poss))
    return output_poss, output_probs

# Given morphemes and their POS, assign the corresponding word the POS of the highest rank.
# The rank is defined in: https://github.com/coastalcph/ud-conversion-tools
POS_PRECEDENCE = ['VERB', 'NOUN', 'PROPN', 'PRON', 'ADJ', 'NUM', 'ADV', 'INTJ', 'AUX', 'ADP', 'DET', 'PART', 'CCONJ', 'SCONJ', 'X', 'SYM', 'PUNCT', '***']
def get_pos_by_precedence(words, morphs, poss, probs):
    word_index = 0
    output_poss = []
    output_probs = []
    current_word = ''
    current_poss = []
    current_probs = []
    for i in range(len(morphs)):
        current_word += morphs[i]
        current_poss.append(poss[i])
        current_probs.append(probs[i])
        if current_word == words[word_index]:
            main_pos, indices = get_main_pos(current_poss)
            output_poss.append(main_pos)
            output_probs.append(max([current_probs[j] for j in indices]))
            word_index += 1
            current_word = ''
            current_poss = []
            current_probs = []
    if len(words) != len(output_poss):
        print(len(words), len(output_poss))
    return output_poss, output_probs

def get_main_pos(pos_list):
    for pos in POS_PRECEDENCE:
        if pos in pos_list:
            indices = [i for i, p in enumerate(pos_list) if p == pos]
            return pos, indices

# Parameters
source_data_path = sys.argv[1]
tagged_morpheme_path = sys.argv[2]
prob_path = sys.argv[3]
stem_path = sys.argv[4]
pos_output_path = sys.argv[5]
prob_output_path = sys.argv[6]

# Read the input files into maps.
word_map = read_file(source_data_path)
morpheme_map = read_file(tagged_morpheme_path)
prob_map = read_file(prob_path)
stem_map = None if stem_path == 'NA' else read_file(stem_path)

# Replace the morphemes and their POS tags by their corresponding words and representative tags.
with open(pos_output_path, 'w', encoding='UTF-8') as pos_fout:
    with open(prob_output_path, 'w', encoding='UTF-8') as prob_fout:
        for key in morpheme_map:
            words = word_map[key].split()
            morphemes = [morphpos.rsplit('_', 1)[0] for morphpos in morpheme_map[key].split()]
            poss = [morphpos.rsplit('_', 1)[1] for morphpos in morpheme_map[key].split()]
            stem_morphemes = None if not stem_map else stem_map[key].split()
            probs = prob_map[key].split()

            output_poss = []
            if stem_morphemes:
                output_poss, output_probs = get_pos_by_stem(words, morphemes, poss, stem_morphemes, probs)
            else:
                output_poss, output_probs = get_pos_by_precedence(words, morphemes, poss, probs)
            pos_fout.write(key+'\t'+' '.join([word+'_'+pos for (word,pos) in zip(words, output_poss)])+'\n')
            prob_fout.write(key+'\t'+' '.join(output_probs)+'\n')