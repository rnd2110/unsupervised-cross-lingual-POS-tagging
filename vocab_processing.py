#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import re

import numpy as np

###############
# CONSTANTS
###############

PAD = "__PAD__"
UNK = "__UNK__"
BLANK = '***'
START = 'START'
END = 'END'
START_MARKER = '<START>'
END_MARKER = '<END>'

REGEX_DIGIT = '[\d٠١٢٣٤٥٦٧٨٩౦౧౨౩౪౫౬౭౮౯፲፳፴፵፶፷፸፹፺፻०१२३४५६७८९४零一二三四五六七八九十百千万億兆つ]'
REGEX_PUNCT1 = r'^[\\\\_\"\“\”\‘\’\``\′\՛\·\.\ㆍ\•\,\、\;\:\?\？\!\[\]\{\}\(\)\|\«\»\…\،\؛\؟\¿\፤\፣\።\፨\፠\፧\፦\፡\…\।\¡\「\」\《\》\』\『\‹\〔\〕\–\—\−\-\„\‚\´\'\〉\〈 \【\】\（\）\~\。\○\．\♪]+$'
REGEX_PUNCT2 = r'^[\*\/\-]{2,}$'
REGEX_SYM1 = r'^[\+\=\≠\%\$\£\€\#\°\@\٪\≤\≥\^\φ\θ\×\✓\✔\△\©\☺\♥\❤]+$'
REGEX_SYM2 = r'^((\:[\)\(DPO])|(\;[\)])|m²)$'
REGEX_SYM3 = r'^'+REGEX_DIGIT+'+(([\.\,\:\-\/])?'+REGEX_DIGIT+')*\%$'
REGEX_EMOJI = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "]+)"
)


REGEX_NUMBER = r'^\%?'+REGEX_DIGIT+'+(([\.\,\:\-\/])?'+REGEX_DIGIT+')*$'

UD_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

UD_PETROV_TAGS = {}
UD_PETROV_TAGS['ADJ'] = 'ADJ'
UD_PETROV_TAGS['ADP'] = 'ADP'
UD_PETROV_TAGS['ADV'] = 'ADV'
UD_PETROV_TAGS['AUX'] = 'VERB'
UD_PETROV_TAGS['CCONJ'] = 'CONJ'
UD_PETROV_TAGS['DET'] = 'DET'
UD_PETROV_TAGS['INTJ'] = 'X'
UD_PETROV_TAGS['NOUN'] = 'NOUN'
UD_PETROV_TAGS['NUM'] = 'NUM'
UD_PETROV_TAGS['PART'] = 'PART'
UD_PETROV_TAGS['PRON'] = 'PRON'
UD_PETROV_TAGS['PROPN'] = 'NOUN'
UD_PETROV_TAGS['PUNCT'] = 'PUNCT'
UD_PETROV_TAGS['SCONJ'] = 'CONJ'
UD_PETROV_TAGS['SYM'] = 'X'
UD_PETROV_TAGS['VERB'] = 'VERB'
UD_PETROV_TAGS['X'] = 'X'


class Vocab:
    def __init__(self, train, tune, tests):

        # Read the words and tags, and create the corresponding maps.

        data = train + tune
        for test in tests:
            data = data + test
        
        token_count, prefix1_count, prefix2_count, prefix3_count, prefix4_count, suffix1_count, suffix2_count, suffix3_count, suffix4_count, characters_count, tags = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), set()
        
        self.max_length = 0

        for token_list, tag_list in data:
        
            for token in token_list:

                token, prefix1, prefix2, prefix3, prefix4, suffix1, suffix2, suffix3, suffix4, chars = self.get_features(token)
                
                token_count[token] += 1
                prefix1_count[prefix1] += 1
                prefix2_count[prefix2] += 1
                prefix3_count[prefix3] += 1
                prefix4_count[prefix4] += 1
                
                suffix1_count[suffix1] += 1
                suffix2_count[suffix2] += 1
                suffix3_count[suffix3] += 1
                suffix4_count[suffix4] += 1

                for character in chars:
                    characters_count[character] += 1
                    
                if len(token) > self.max_length:
                    self.max_length = len(token)

            for tag in tag_list:
                tags.add(tag)
                
        tags = list(tags)
        if BLANK in tags:
            tags.remove(BLANK)
        if START in tags:
            tags.remove(START)
        if END in tags:
            tags.remove(END)

        missing_tags = []
        for ud_tag in UD_TAGS:
            if ud_tag not in tags:
                missing_tags.append(ud_tag)
                tags.append(ud_tag)
        print("Missing tags: " + str(missing_tags))

        self.tokens = [UNK] + list(token_count.keys())

        self.prefixes1 = [UNK] + list(prefix1_count.keys())
        self.prefixes2 = [UNK] + list(prefix2_count.keys())
        self.prefixes3 = [UNK] + list(prefix3_count.keys())
        self.prefixes4 = [UNK] + list(prefix4_count.keys())

        self.suffixes1 = [UNK] + list(suffix1_count.keys())
        self.suffixes2 = [UNK] + list(suffix2_count.keys())
        self.suffixes3 = [UNK] + list(suffix3_count.keys())
        self.suffixes4 = [UNK] + list(suffix4_count.keys())

        self.characters = [UNK] + list(characters_count.keys())

        self.output_tags = [BLANK] + tags + [START] + [END]

        self.token_dict = {token: i for i, token in enumerate(self.tokens)}

        self.prefix1_dict = {prefix: i for i, prefix in enumerate(self.prefixes1)}
        self.prefix2_dict = {prefix: i for i, prefix in enumerate(self.prefixes2)}
        self.prefix3_dict = {prefix: i for i, prefix in enumerate(self.prefixes3)}
        self.prefix4_dict = {prefix: i for i, prefix in enumerate(self.prefixes4)}

        self.suffix1_dict = {suffix: i for i, suffix in enumerate(self.suffixes1)}
        self.suffix2_dict = {suffix: i for i, suffix in enumerate(self.suffixes2)}
        self.suffix3_dict = {suffix: i for i, suffix in enumerate(self.suffixes3)}
        self.suffix4_dict = {suffix: i for i, suffix in enumerate(self.suffixes4)}

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
                tag2 = self.tag2id(tag_list[i+1])
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
                    
    def token2id(self, token):
        return self.token_dict[token] if token in self.token_dict else self.token_dict[UNK]

    def id2token(self, id_):
        return self.tokens[id_] if id_ < len(self.tokens) else UNK

    def allowed_tags(self, token):
        return self.token_tags[token] if (token in self.token_tags and self.tag2id(BLANK) not in self.token_tags[token]) else range(len(self.output_tags))

    def prefix12id(self, prefix):
        return self.prefix1_dict[prefix] if prefix in self.prefix1_dict else self.prefix1_dict[UNK]

    def prefix22id(self, prefix):
        return self.prefix2_dict[prefix] if prefix in self.prefix2_dict else self.prefix2_dict[UNK]

    def prefix32id(self, prefix):
        return self.prefix3_dict[prefix] if prefix in self.prefix3_dict else self.prefix3_dict[UNK]

    def prefix42id(self, prefix):
        return self.prefix4_dict[prefix] if prefix in self.prefix4_dict else self.prefix4_dict[UNK]

    def suffix12id(self, suffix):
        return self.suffix1_dict[suffix] if suffix in self.suffix1_dict else self.suffix1_dict[UNK]

    def suffix22id(self, suffix):
        return self.suffix2_dict[suffix] if suffix in self.suffix2_dict else self.suffix2_dict[UNK]

    def suffix32id(self, suffix):
        return self.suffix3_dict[suffix] if suffix in self.suffix3_dict else self.suffix3_dict[UNK]

    def suffix42id(self, suffix):
        return self.suffix4_dict[suffix] if suffix in self.suffix4_dict else self.suffix4_dict[UNK]

    def character2id(self, character):
        return self.characters_dict[character] if character in self.characters_dict else self.characters_dict[UNK]

    def id2tag(self, id_):
        return self.output_tags[id_] if id_ < len(self.output_tags) else UNK

    def tag2id(self, tag):
        return self.output_tag_dict[tag] if tag in self.output_tag_dict else -1

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

    def num_suffixes1(self):
        return len(self.suffixes1)

    def num_suffixes2(self):
        return len(self.suffixes2)

    def num_suffixes3(self):
        return len(self.suffixes3)

    def num_suffixes4(self):
        return len(self.suffixes4)
    
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
            return START_MARKER, START_MARKER, START_MARKER, START_MARKER, START_MARKER, START_MARKER, START_MARKER, START_MARKER, START_MARKER, [START_MARKER for i in range(self.max_length)]
        elif token == END_MARKER:
            return END_MARKER, END_MARKER, END_MARKER, END_MARKER, END_MARKER, END_MARKER, END_MARKER, END_MARKER, END_MARKER, [END_MARKER for i in range(self.max_length)]
        
        token = simplify_token(token)
        prefix1 = token[0:1]
        prefix2 = token[0:2] if len(token) > 1 else prefix1
        prefix3 = token[0:3] if len(token) > 2 else prefix2
        prefix4 = token[0:4] if len(token) > 3 else prefix3
        suffix1 = token[len(token)-1:len(token)]
        suffix2 = token[len(token)-2:len(token)] if len(token) > 1 else suffix1
        suffix3 = token[len(token)-3:len(token)] if len(token) > 2 else suffix2
        suffix4 = token[len(token)-4:len(token)] if len(token) > 3 else suffix3
        chars = list(token)
        for i in range(self.max_length - len(token)):
            chars.append(PAD)
        return token, prefix1, prefix2, prefix3, prefix4, suffix1, suffix2, suffix3, suffix4, chars

    # Postprocess the tags (some linguistics here...)
    def postprocess_tags(self, tags, tokens):
        postprocessed_tags = []
        for i in range(len(tokens)):
            postprocessed_tag_label = ''
            if re.match(REGEX_SYM1, tokens[i]) or re.match(REGEX_SYM2, tokens[i]) or re.match(REGEX_SYM3, tokens[i]) or re.match(REGEX_EMOJI, tokens[i]):
                postprocessed_tag_label = 'SYM'
            elif re.match(REGEX_PUNCT1, tokens[i]) or re.match(REGEX_PUNCT2, tokens[i]):
                postprocessed_tag_label = 'PUNCT'
            elif re.match(REGEX_NUMBER, tokens[i]):
                postprocessed_tag_label = 'NUM'
            predicted_tag_label = self.id2tag(int(tags[i]))
            if predicted_tag_label == 'PUNCT' and postprocessed_tag_label == '':
                if re.match('.*[A-Z].*', tokens[i]) and tokens[i].upper() == tokens[i]:
                    if re.match('^[XIV\.]{2,}$', tokens[i]):
                        postprocessed_tag_label = 'NUM'
                    else:
                        postprocessed_tag_label = 'PROPN'
                else:
                    postprocessed_tag_label = 'NOUN'  # default
            if postprocessed_tag_label == '':
                postprocessed_tag_label = predicted_tag_label
            postprocessed_tags.append(int(self.tag2id(postprocessed_tag_label)))
        return postprocessed_tags

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
            if test_data_set is not None and (test_data_set == 'DEV20' or test_data_set == 'DEV22' or test_data_set == 'TEST20' or test_data_set == 'TEST22'):
                if predicted_tag_label == 'INTJ':
                    predicted_tag = self.tag2id('CCONJ')
            if test_data_set is not None and (test_data_set == 'DEV20' or test_data_set == 'TEST20'):
                if predicted_tag_label == 'DET':
                    predicted_tag = self.tag2id('ADJ')
        elif target_language == 'PER':
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
        return self.tag2id(UD_PETROV_TAGS[predicted_tag_label]) if predicted_tag_label in UD_PETROV_TAGS else predicted_tag

# Simply a given token by replacing the digits/numbers with zeros.
def simplify_token(token):
    chars = []
    for char in token:
        #### Reduce sparsity by replacing all digits with 0.
        if char.isdigit()  or re.match(REGEX_DIGIT, char) or (re.match('[零一二三四五六七八九十百千万億兆つ]', char) and re.match(REGEX_NUMBER, token)):
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)
