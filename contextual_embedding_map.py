#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import ast

CONTEXTUAL_EMBEDDING_DIM = 1280

class Contextual_Embedding_Map:

    def __init__(self, contextual_embedding_paths, contextual_tokenization_paths, subword_combination_method, sentences):

        self.contextual_embeddings = {}
        self.max_contextual_embedding_value = 1.0;
        self.contextual_embedding_paths = contextual_embedding_paths
        self.contextual_tokenization_paths = contextual_tokenization_paths
        self.subword_combination_method = subword_combination_method
        self.sentences = sentences
 
    def load_embeddings(self):

        tokenization = {}

        #### Read tokenization information
        contextual_tokenization_path_list = self.contextual_tokenization_paths.split(',')
        for contextual_tokenization_path in contextual_tokenization_path_list:
            with open(contextual_tokenization_path) as tokenization_lines:
                for line in tokenization_lines:
                    line = line.rstrip()
                    if len(line) == 0:
                        continue;
                    splits = line.split('\t')
                    if len(splits) > 1:
                        tokenization[splits[0]] = splits[1].split()

        self.contextual_embeddings = {}
        self.max_contextual_embedding_value = 1.0
        sentence_index = 0

        #### Read word embeddings
        contextual_embedding_path_list = self.contextual_embedding_paths.split(',')
        for contextual_embedding_path in contextual_embedding_path_list:
            with open(contextual_embedding_path) as embedding_lines:
                status = 0
                sentence = ''
                tokens = []
                token_index = 0
                embeddings = []
                for line in embedding_lines:
                    line = line.rstrip()
                    if status == 0:
                        sentence = line
                        status = 1
                    elif status == 1:
                        tokens = ast.literal_eval(line)[1:-1]
                        status = 2
                    elif status == 2:
                        token_index = token_index + 1
                        if sentence not in self.sentences:
                            if token_index == len(tokens):
                                status = 0
                                sentence = ''
                                tokens = []
                                token_index = 0
                                embeddings = []
                            continue
                        current_word_embeddings = np.fromstring(line, dtype=float, sep=", ")
                        embeddings.append(current_word_embeddings)
                        processed_embeddings = []
                        if token_index == len(tokens):
                            current_token_index = 0
                            sentence_tokens = sentence.split(' ')
                            for sentence_token in sentence_tokens:
                                ending_index = current_token_index + len(tokenization[sentence_token])
                                subwords = tokens[current_token_index: ending_index]
                                current_embeddings = embeddings[current_token_index: ending_index]

                                token_embedding = []
                                if self.subword_combination_method == 'AVERAGE':
                                    token_embedding = self.get_embedding_average(current_embeddings)
                                elif self.subword_combination_method == 'FIRST':
                                    token_embedding = self.get_embedding_first(current_embeddings)
                                elif self.subword_combination_method == 'FIRST_LAST':
                                    token_embedding = self.get_embedding_first_last(current_embeddings)
                                elif self.subword_combination_method == 'LONGEST':
                                    token_embedding = self.get_embedding_longest(subwords, current_embeddings)

                                max_value = np.amax(token_embedding)
                                min_value = np.amin(token_embedding)
                                if max_value > self.max_contextual_embedding_value:
                                    self.max_contextual_embedding_value = max_value
                                if -1 * min_value > self.max_contextual_embedding_value:
                                    self.max_contextual_embedding_value = -1 * min_value

                                processed_embeddings.append(token_embedding)
                                current_token_index = ending_index
                            self.contextual_embeddings[sentence] = processed_embeddings
                            if len(sentence.split()) != len(processed_embeddings):
                                print(sentence)
                            sentence_index = sentence_index+1
                            if sentence_index%100000 == 0:
                                print(sentence_index)
                            status = 0
                            sentence = ''
                            tokens = []
                            token_index = 0
                            embeddings = []

        print("Contextual-embedding Sentence Count: ", len(self.contextual_embeddings))

    def get_embedding_average(self, embeddings):
        return np.average(embeddings, axis=0)

    def get_embedding_first(self, embeddings):
        return embeddings[0]

    def get_embedding_first_last(self, embeddings):
        if len(embeddings) == 1:
            return embeddings[0]
        fl_embeddings = []
        fl_embeddings.append(embeddings[0])
        fl_embeddings.append(embeddings[-1])
        return self.get_embedding_average(fl_embeddings)

    def get_embedding_longest(self, subwords, embeddings):
        max_index = 0
        max_len = -1
        for i in range(len(subwords)):
            subword_length = len(subwords[i].replace("@@", ""))
            if subword_length > max_len:
                max_index = i
                max_len = subword_length
        return embeddings[max_index]
