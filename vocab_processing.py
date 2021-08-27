import numpy as np

from trivial_pos import *
from segmentation import *
from utils import *

# CONSTANTS

PAD = "__PAD__"
UNK = "__UNK__"
BLANK = '***'
START = 'START'
END = 'END'
START_MARKER = '<START>'
END_MARKER = '<END>'

PREFIX_SPLIT = 'PrefixMorph'
STEM_SPLIT = 'Stem'
SUFFIX_SPLIT = 'SuffixMorph'
MIN_SEGMENTATION_WORD_LENGTH = 3

UD_TAGS = [BLANK, 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
           'SCONJ', 'SYM', 'VERB', 'X', START, END]
UD_PETROV_TAGS = {'AUX':'VERB', 'CCONJ':'CONJ', 'INTJ':'X', 'PROPN':'NOUN', 'SCONJ':'CONJ', 'SYM':'X'}

FORCED_NOUN_PROB = 0.33333333
FORCED_PROPN_PROB = 0.16666667

class Vocab:
    def __init__(self, language, train, tune, tests, segmentation_grammar_output_path):

        # Prepare the segmentation model.
        self.language = language
        self.segmentation_model = None
        if segmentation_grammar_output_path.upper() != 'NA':
            self.segmentation_model = parse_segmentation_output(segmentation_grammar_output_path, PREFIX_SPLIT,
                                                                STEM_SPLIT, SUFFIX_SPLIT, None, language,
                                                                MIN_SEGMENTATION_WORD_LENGTH)

        # Read the words and tags, and create the corresponding maps.
        data = train + tune
        for test in tests:
            data = data + test

        token_count, prefix1_count, prefix2_count, prefix3_count, prefix4_count, suffix1_count, suffix2_count, suffix3_count, suffix4_count, characters_count, complex_prefix_count, prefix_count, stem_count, complex_suffix_count, suffix_count, tags = defaultdict(
            int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(
            int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(
            int), defaultdict(int), defaultdict(int), set()
        complex_prefix_count[''] = 0
        prefix_count[''] = 0
        complex_suffix_count[''] = 0
        suffix_count[''] = 0

        self.max_length = 0

        for token_list, tag_list in data:

            for token_index, token in enumerate(token_list):

                # Get token basic features.
                token, prefix1, prefix2, prefix3, prefix4, complex_prefix, prefixes, stem, suffix1, suffix2, suffix3, suffix4, complex_suffix, suffixes, chars = self.get_features(token)

                token_count[token] += 1
                prefix1_count[prefix1] += 1
                prefix2_count[prefix2] += 1
                prefix3_count[prefix3] += 1
                prefix4_count[prefix4] += 1
                complex_prefix_count[complex_prefix] += 1
                for prefix in prefixes:
                    prefix_count[prefix] += 1
                stem_count[stem] += 1
                suffix1_count[suffix1] += 1
                suffix2_count[suffix2] += 1
                suffix3_count[suffix3] += 1
                suffix4_count[suffix4] += 1
                complex_suffix_count[complex_suffix] += 1
                for suffix in suffixes:
                    suffix_count[suffix] += 1

                for character in chars:
                    characters_count[character] += 1

                if len(token) > self.max_length:
                    self.max_length = len(token)

        # Add UNK values.
        self.tokens = [UNK] + list(token_count.keys())
        self.prefixes1 = [UNK] + list(prefix1_count.keys())
        self.prefixes2 = [UNK] + list(prefix2_count.keys())
        self.prefixes3 = [UNK] + list(prefix3_count.keys())
        self.prefixes4 = [UNK] + list(prefix4_count.keys())
        self.complex_prefixes = [UNK] + list(complex_prefix_count.keys())
        self.prefixes = [UNK] + list(prefix_count.keys())
        self.stems = [UNK] + list(stem_count.keys())
        self.suffixes1 = [UNK] + list(suffix1_count.keys())
        self.suffixes2 = [UNK] + list(suffix2_count.keys())
        self.suffixes3 = [UNK] + list(suffix3_count.keys())
        self.suffixes4 = [UNK] + list(suffix4_count.keys())
        self.complex_suffixes = [UNK] + list(complex_suffix_count.keys())
        self.suffixes = [UNK] + list(suffix_count.keys())
        self.characters = [UNK] + list(characters_count.keys())
        self.output_tags = UD_TAGS

        # Build lookups.
        self.token_dict = {token: i for i, token in enumerate(self.tokens)}
        self.prefix1_dict = {prefix: i for i, prefix in enumerate(self.prefixes1)}
        self.prefix2_dict = {prefix: i for i, prefix in enumerate(self.prefixes2)}
        self.prefix3_dict = {prefix: i for i, prefix in enumerate(self.prefixes3)}
        self.prefix4_dict = {prefix: i for i, prefix in enumerate(self.prefixes4)}
        self.complex_prefix_dict = {prefix: i for i, prefix in enumerate(self.complex_prefixes)}
        self.prefix_dict = {prefix: i for i, prefix in enumerate(self.prefixes)}
        self.stem_dict = {stem: i for i, stem in enumerate(self.stems)}
        self.suffix1_dict = {suffix: i for i, suffix in enumerate(self.suffixes1)}
        self.suffix2_dict = {suffix: i for i, suffix in enumerate(self.suffixes2)}
        self.suffix3_dict = {suffix: i for i, suffix in enumerate(self.suffixes3)}
        self.suffix4_dict = {suffix: i for i, suffix in enumerate(self.suffixes4)}
        self.complex_suffix_dict = {suffix: i for i, suffix in enumerate(self.complex_suffixes)}
        self.suffix_dict = {suffix: i for i, suffix in enumerate(self.suffixes)}
        self.characters_dict = {character: i for i, character in enumerate(self.characters)}
        self.output_tag_dict = {tag: i for i, tag in enumerate(self.output_tags)}

        # Extract the training vocabulary.
        print("Extracting training vocabulary...")
        self.in_vocab = set()
        for token_list, _ in train:
            for i in range(len(token_list)):
                self.in_vocab.add(simplify_token(token_list[i]))

        # Calculate the transition matrix (not currently used).
        print("Calculating transition matrix...")
        self.transitions = [[0 for x in range(len(self.output_tags))] for y in range(len(self.output_tags))]
        total = 0
        for _, tag_list in train:
            for i in range(len(tag_list) - 1):
                tag1 = self.tag2id(tag_list[i])
                tag2 = self.tag2id(tag_list[i + 1])
                self.transitions[tag1][tag2] = self.transitions[tag1][tag2] + 1
                total = total + 1
        summ = np.sum(self.transitions, axis=1)
        for i in range(len(self.transitions)):
            for j in range(len(self.transitions[i])):
                self.transitions[i][j] = 0 if summ[i] == 0 else self.transitions[i][j] / summ[i]

        # Calculate the type constraints (not currently used).
        print("Getting type constraints...")
        self.token_tags = {}
        for token_list, tag_list in train:
            for i in range(len(token_list)):
                token = self.token2id(simplify_token(token_list[i]))
                tag = self.tag2id(tag_list[i])
                if not token in self.token_tags:
                    self.token_tags[token] = []
                if not tag in self.token_tags[token]:
                    self.token_tags[token].append(tag)

    # Getters
    def token2id(self, token):
        return self.token_dict[token] if token in self.token_dict else self.token_dict[UNK]

    def id2token(self, id_):
        return self.tokens[id_] if id_ < len(self.tokens) else UNK

    def allowed_tags(self, token):
        return self.token_tags[token] if (
                    token in self.token_tags and self.tag2id(BLANK) not in self.token_tags[token]) else range(
            len(self.output_tags))

    def prefix12id(self, prefix):
        return self.prefix1_dict[prefix] if prefix in self.prefix1_dict else self.prefix1_dict[UNK]

    def prefix22id(self, prefix):
        return self.prefix2_dict[prefix] if prefix in self.prefix2_dict else self.prefix2_dict[UNK]

    def prefix32id(self, prefix):
        return self.prefix3_dict[prefix] if prefix in self.prefix3_dict else self.prefix3_dict[UNK]

    def prefix42id(self, prefix):
        return self.prefix4_dict[prefix] if prefix in self.prefix4_dict else self.prefix4_dict[UNK]

    def complex_prefix2id(self, prefix):
        return self.complex_prefix_dict[prefix] if prefix in self.complex_prefix_dict else self.complex_prefix_dict[UNK]

    def prefix2id(self, prefix):
        return self.prefix_dict[prefix] if prefix in self.prefix_dict else self.prefix_dict[UNK]

    def stem2id(self, stem):
        return self.stem_dict[stem] if stem in self.stem_dict else self.stem_dict[UNK]

    def suffix12id(self, suffix):
        return self.suffix1_dict[suffix] if suffix in self.suffix1_dict else self.suffix1_dict[UNK]

    def suffix22id(self, suffix):
        return self.suffix2_dict[suffix] if suffix in self.suffix2_dict else self.suffix2_dict[UNK]

    def suffix32id(self, suffix):
        return self.suffix3_dict[suffix] if suffix in self.suffix3_dict else self.suffix3_dict[UNK]

    def suffix42id(self, suffix):
        return self.suffix4_dict[suffix] if suffix in self.suffix4_dict else self.suffix4_dict[UNK]

    def complex_suffix2id(self, suffix):
        return self.complex_suffix_dict[suffix] if suffix in self.complex_suffix_dict else self.complex_suffix_dict[UNK]

    def suffix2id(self, suffix):
        return self.suffix_dict[suffix] if suffix in self.suffix_dict else self.suffix_dict[UNK]

    def character2id(self, character):
        return self.characters_dict[character] if character in self.characters_dict else self.characters_dict[UNK]

    def id2tag(self, id_):
        return self.output_tags[id_] if (id_ >= 0 and id_ < len(self.output_tags)) else UNK

    def tag2id(self, tag):
        return self.output_tags.index(tag) if tag in self.output_tags else -1

    def num_tokens(self):
        return len(self.tokens)

    def num_prefixes1(self):
        return len(self.prefixes1)

    def num_prefixes2(self):
        return len(self.prefixes2)

    def num_prefixes3(self):
        return len(self.prefixes3)

    def num_prefixes4(self):
        return len(self.prefixes4)

    def num_complex_prefixes(self):
        return len(self.complex_prefixes)

    def num_prefixes(self):
        return len(self.prefixes)

    def num_stems(self):
        return len(self.stems)

    def num_suffixes1(self):
        return len(self.suffixes1)

    def num_suffixes2(self):
        return len(self.suffixes2)

    def num_suffixes3(self):
        return len(self.suffixes3)

    def num_suffixes4(self):
        return len(self.suffixes4)

    def num_complex_suffixes(self):
        return len(self.complex_suffixes)

    def num_suffixes(self):
        return len(self.suffixes)

    def num_characters(self):
        return len(self.characters)

    def max_token_length(self):
        return self.max_length

    def num_tags(self):
        return len(self.output_tags)

    def blank_tag_id(self):
        return self.tag2id(BLANK)

    def start_tag_id(self):
        return self.tag2id(START)

    def end_tag_id(self):
        return self.tag2id(END)

    def is_IV(self, token):
        return token in self.in_vocab

    # Extract the features for a given token.
    def get_features(self, token):
        if token == START_MARKER:
            return START_MARKER, START_MARKER, START_MARKER, START_MARKER, START_MARKER, START_MARKER, [
                START_MARKER], START_MARKER, START_MARKER, START_MARKER, START_MARKER, START_MARKER, START_MARKER, [
                       START_MARKER], [START_MARKER for i in range(self.max_length)]
        elif token == END_MARKER:
            return END_MARKER, END_MARKER, END_MARKER, END_MARKER, END_MARKER, END_MARKER, [
                END_MARKER], END_MARKER, END_MARKER, END_MARKER, END_MARKER, END_MARKER, END_MARKER, [END_MARKER], [
                       END_MARKER for i in range(self.max_length)]

        token = simplify_token(token)
        prefix1 = token[0:1]
        prefix2 = token[0:2] if len(token) > 1 else prefix1
        prefix3 = token[0:3] if len(token) > 2 else prefix2
        prefix4 = token[0:4] if len(token) > 3 else prefix3
        suffix1 = token[len(token) - 1:len(token)]
        suffix2 = token[len(token) - 2:len(token)] if len(token) > 1 else suffix1
        suffix3 = token[len(token) - 3:len(token)] if len(token) > 2 else suffix2
        suffix4 = token[len(token) - 4:len(token)] if len(token) > 3 else suffix3
        chars = list(token)
        for i in range(self.max_length - len(token)):
            chars.append(PAD)

        segmented_token = segment_text(token, self.segmentation_model, ' ', '@@', False, self.language,
                                       MIN_SEGMENTATION_WORD_LENGTH) if self.segmentation_model else None
        splits = segmented_token.split('@@') if segmented_token else None
        complex_prefix = splits[0].replace(' ', '') if splits else ''
        prefixes = splits[0].split() if splits else []
        stem = splits[1] if splits else ''
        complex_suffix = splits[2].replace(' ', '') if splits else ''
        suffixes = splits[2].split() if splits else []

        return token, prefix1, prefix2, prefix3, prefix4, complex_prefix, prefixes, stem, suffix1, suffix2, suffix3, suffix4, complex_suffix, suffixes, chars

    # Postprocess the tags (some linguistics here...).
    def postprocess_tags(self, tags, tokens, prob):
        postprocessed_tags = []
        for i in range(len(tokens)):
            postprocessed_tag_label = ''
            if is_symbol(tokens[i]):
                postprocessed_tag_label = 'SYM'
                prob[i] = 1.0
            elif is_punctuation(tokens[i]):
                postprocessed_tag_label = 'PUNCT'
                prob[i] = 1.0
            elif is_number(tokens[i]):
                postprocessed_tag_label = 'NUM'
                prob[i] = 1.0
            predicted_tag_label = self.id2tag(int(tags[i]))
            if predicted_tag_label == 'PUNCT' and postprocessed_tag_label == '':
                if re.match('.*[A-Z].*', tokens[i]) and tokens[i].upper() == tokens[i]:
                    if re.match('^[XIV\.]{2,}$', tokens[i]):
                        postprocessed_tag_label = 'NUM'
                        prob[i] = 1.0
                    else:
                        postprocessed_tag_label = 'PROPN'
                        prob[i] = FORCED_PROPN_PROB
                else:
                    postprocessed_tag_label = 'NOUN'  # default
                    prob[i] = FORCED_NOUN_PROB
            if postprocessed_tag_label == '':
                postprocessed_tag_label = predicted_tag_label
            postprocessed_tags.append(int(self.tag2id(postprocessed_tag_label)))
        return postprocessed_tags, prob

    # Fix the tags for the underlying language following the UD annotations.
    def fix_tag(self, target_language, predicted_tag, test_data_set):
        predicted_tag_label = self.id2tag(predicted_tag)
        if target_language == 'AFR':
            if predicted_tag_label == 'INTJ':
                predicted_tag = self.tag2id('PART')
        elif target_language == 'AMH':
            if predicted_tag_label == 'SYM':
                predicted_tag = self.tag2id('PUNCT')
            elif predicted_tag_label == 'X':
                predicted_tag = self.tag2id('NOUN')
        elif target_language == 'BUL':
            if predicted_tag_label == 'SYM':
                predicted_tag = self.tag2id('NOUN')
        elif target_language == 'EUS':
            if predicted_tag_label == 'SCONJ':
                predicted_tag = self.tag2id('CCONJ')
        elif target_language == 'FIN':
            if predicted_tag_label == 'DET':
                predicted_tag = self.tag2id('PRON')
            elif predicted_tag_label == 'PART':
                predicted_tag = self.tag2id('ADV')
        elif target_language == 'HIN':
            if predicted_tag_label == 'SYM':
                predicted_tag = self.tag2id('PUNCT')
        elif target_language == 'IND':
            if predicted_tag_label == 'INTJ':
                predicted_tag = self.tag2id('PART')
            if test_data_set is not None and (test_data_set == 'DEV12' or test_data_set == 'TEST12'):
                if predicted_tag_label == 'AUX':
                    predicted_tag = self.tag2id('VERB')
        elif target_language == 'JPN1':
            if predicted_tag_label == 'X':
                predicted_tag = self.tag2id('NOUN')
            if test_data_set is not None and (
                    test_data_set == 'DEV20' or test_data_set == 'DEV22' or test_data_set == 'TEST20' or test_data_set == 'TEST22'):
                if predicted_tag_label == 'INTJ':
                    predicted_tag = self.tag2id('CCONJ')
            if test_data_set is not None and (test_data_set == 'DEV20' or test_data_set == 'TEST20'):
                if predicted_tag_label == 'DET':
                    predicted_tag = self.tag2id('ADJ')
        elif target_language == 'PES':
            if predicted_tag_label == 'SYM':
                predicted_tag = self.tag2id('PUNCT')
            elif predicted_tag_label == 'PROPN':
                predicted_tag = self.tag2id('NOUN')
        elif target_language == 'POR':
            if test_data_set is not None and (test_data_set == 'DEV12' or test_data_set == 'TEST12'):
                if predicted_tag_label == 'SYM':
                    predicted_tag = self.tag2id('NOUN')
        elif target_language == 'TEL':
            if predicted_tag_label == 'AUX':
                predicted_tag = self.tag2id('VERB')
            elif predicted_tag_label == 'SYM':
                predicted_tag = self.tag2id('PUNCT')
            elif predicted_tag_label == 'X':
                predicted_tag = self.tag2id('NOUN')
        elif target_language == 'TUR':
            if predicted_tag_label == 'PART':
                predicted_tag = self.tag2id('ADV')
            elif predicted_tag_label == 'SCONJ':
                predicted_tag = self.tag2id('ADP')
            elif predicted_tag_label == 'SYM':
                predicted_tag = self.tag2id('PUNCT')
        return int(predicted_tag)

    def convert_tag_to_petrov12(self, predicted_tag):
        predicted_tag_label = self.id2tag(predicted_tag)
        return self.tag2id(
            UD_PETROV_TAGS[predicted_tag_label]) if predicted_tag_label in UD_PETROV_TAGS else predicted_tag


# Simplify a given token by replacing the digits/numbers with zeros.
def simplify_token(token):
    chars = []
    for char in token:
        # Reduce sparsity by replacing all digits with 0.
        if char.isdigit() or is_number(char) or (is_number(char) and is_number(token)):
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)