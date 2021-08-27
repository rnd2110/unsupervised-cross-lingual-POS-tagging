#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

# REGEX for numbers, punctuation marks and symbols
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

# Checking for trivial POS types
def is_number(token):
    return re.match(REGEX_NUMBER, token)

def is_punctuation(token):
    return re.match(REGEX_PUNCT1, token) or re.match(REGEX_PUNCT2, token)

def is_symbol(token):
    return re.match(REGEX_SYM1, token) or re.match(REGEX_SYM2, token) or re.match(REGEX_SYM3, token) or re.match(REGEX_EMOJI, token)

def is_trivial(token):
    return is_number(token) or is_punctuation(token) or is_symbol(token)