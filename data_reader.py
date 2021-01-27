#!/usr/bin/env python
# -*- coding: utf-8 -*-

from vocab_processing import *

HELDOUT_PERCENTAGE = 10

def read_data(train, target_language, source_language, data_path, data_set, training_size, max_sentence_length, min_density, hold_out):
    data_path = get_data_file_path(train, target_language, source_language, data_path, data_set)
    content = []
    held_out_content = []
    index = 0
    word_count = 0

    #### Read the data
    with open(data_path) as data_src:
        for line in data_src:
            all_count = line.count(' ')+1
            blank_count = line.count(BLANK)
            percentage = 1 - blank_count / all_count

            if percentage < min_density:
                continue

            t_p = [w.rsplit("_", 1) for w in line.strip().split()]
            tokens = [v[0] for v in t_p]
            if len(tokens) > max_sentence_length and max_sentence_length > 0:
                continue

            word_count += len(tokens)
            if word_count > training_size and training_size != -1:
                break

            tokens.insert(0, START_MARKER)
            tokens.append(END_MARKER)
            tags = [v[1] for v in t_p]
            tags.insert(0, START)
            tags.append(END)
            if hold_out and ((index+1) % HELDOUT_PERCENTAGE) == 0:
                held_out_content.append((tokens, tags))
                content.append((tokens, tags))
            else:
                content.append((tokens, tags))
            index = index + 1

    return content, held_out_content

def get_data_file_path(train, target_language, source_language, data_path, data_set):
    if train:
        if source_language != target_language:
            return data_path + '/' + target_language + '-' + source_language + '-bible-POSUD-' + data_set + '.txt'
        else:
            return data_path + '/' + target_language + '-UD-POSUD-' + data_set + '.txt'
    else:
        return data_path + '/' + target_language + '-UD-POSUD-' + data_set + '.txt'
