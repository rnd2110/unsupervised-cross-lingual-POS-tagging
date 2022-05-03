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

# Input parameters
source_data_path = sys.argv[1]
tagged_stem_path = sys.argv[2]
pos_output_path = sys.argv[3]

# Read the input files into maps.
source = read_file(source_data_path)
target = read_file(tagged_stem_path)

# Replace the stems by their corresponding words and tags.
with open(pos_output_path, 'w', encoding='UTF-8') as fout:
    for key in target:
        output_words = source[key].split()
        output_poss = [wordpos.rsplit('_', 1)[1] for wordpos in target[key].split()]
        fout.write(key+'\t'+' '.join([word+'_'+pos for (word, pos) in zip(output_words, output_poss)])+'\n')
