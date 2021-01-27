##  Unsupervised Cross-Lingual Part-of-Speech Tagging ##
##### Version: 1.0
<br/>

### Publications

- [Unsupervised Cross-Lingual Part-of-Speech Tagging for Truly Low-Resource Scenarios](https://www.aclweb.org/anthology/2020.emnlp-main.391.pdf "Unsupervised Cross-Lingual Part-of-Speech Tagging for Truly Low-Resource Scenarios")

---

### Requirements

- [Python 3.x+](https://www.python.org/downloads/ "Python 3.x+")
- [PyTorch (>1.1)]( https://pytorch.org/get-started/locally/ "PyTorch (>1.1)")
- [NumPy](https://numpy.org/ "NumPy")

---

### Trainign and Testing

The main script `tagger.py` is responsible for training and testing the neural POS tagger in one fell swoop. However, it should be straightforward to split the training and testing phases, if needed.

Please note that the current published version requires the fully/partially annotated data as an input. The repo will be updated to add the code for cross-lingual POS alignment soon!
<br/>

#### Main Script: tagger.py

#### Parameters
-  **target_language**: the ISO3 code of the target language
-  **source_language**: the ISO3 code of the source language
- **data_path**: the path of the training and testing data (directory)
- **output_path**: the path of the final tagged output (directory)
- **model_path**: the path of the final model (directory)
- **training_data_set**: the name of the training dataset
- **test_data_sets**: the name(s) of the test dataset(s), comma-separated
- **training_size**: the number of words to train on, -1 = all
- **max_sentence_length**: the maximum sentence length (in words) to train on, -1 = all
- **min_density**: the percentage of partially tagged words to the number of words in a sentence to train on, -1 = all
- **use_word_embeddings**: boolean: whether to use randomly initialized word  embeddings (recommended)
- **use_affix_embeddings**: whether to use randomly initialized prefix/suffix (of lengths 1, 2, 3 and 4) embeddings (recommended).
- **use_char_embeddings**: whether to use character embeddings (not recommended)
- **use_brown_clusters**: whether to use Brown clusters (recommended)
- **brown_cluster_path**: the path of Brown clusters
- **use_contextual_embeddings**: whether to use contextual embeddings (e.g., BERT or XLM) (recommended)
- **contextual_embeddings_dimensions**: vector size for contextual embeddings
- **contextual_embedding_path**: the path of the precomputed contextual embeddings (see the description under the *Notes* section)
- **contextual_tokenization_path**: the path of the tokenization file, a tabular file of two columns: words and space-separated tokens
- **subword_combination_method**: how to combine the embeddings of subwords; the values are: *AVERAGE*, *FIRST*, *FIRST_LAST* and *LONGEST* (recommended: *FIRST_LAST*)
- **epochs**: number of epochs (recommended: 12)
- **learning_rate**: learning rate (ecommended: 0.0001)
- **learning_decay_rate**: learning decay rate (recommended: 0.1)
- **dropout_rate**: dropout rate (recommended: 0.7)
- **fix_tags**: whether to match the output tags with the UD annotation guidelines for the underlying language (e.g., converting PRT to ADV in TUR)
- **run_postprocessing**: whether to force rule-based tagging for punctuation marks, symbols and numbers in the output (recommended)
- **overwrite_by_output**: whether to  use the tags in the test dataset(s) (e.g., when partially annotated) to overwrite system output

#### Notes
- The system assumes all the contextual embeddings are precomputed. However, it is straightforward to change this into runtime computations, if needed.
- In the embeddings file, each sentence should occupy n+2 lines. The first line contains the white-separated tokenized text; the second line contains a vector of subwords or subword IDs, while the *(n+2)th* line contains the comma-separated vector of the *nth* token.
Example
`Eta haur eçagut cedin Ioppe gucian eta sinhets ceçaten anhitzec Iauna baithan .`
['[CLS]', 25623, 56155, 28, 2968, 13306, 405, 3035, 28136, 7340, 2497, 69438, 522, 3811, 7831, 405, 2968, 510, 3616, 24374, 3240, 12044, 1946, 63670, 1121, 6, 5, '[SEP]']`
0.6276292204856873, -0.8384165167808533, 0.6102157235145569, -0.2547730505466461, -0.45138606429100037,.....`
- The training dataset should be named as *(target_language)-(source_language)-POSUD-(training_data_set).txt*, e.g., *EUS-ENG-POSUD-TRAIN.txt*.
- The test dataset(s) should be named as  *(target_language)-(source_language)-POSUD-(test_data_sets).txt*, e.g., *EUS-ENG-POSUD-TEST.txt*.
- The training and testing datasets should have one sentence per line, where each word is represented as *word_POS*, and empty tags are marked as \*\*\*. 
Example:  `Deur_ADP saam_*** te_PART werk_VERB ,_PUNCT kan_*** ons_PRON meer_DET bereik_VERB ._PUNCT`
- We use the output of the Brown-Clustering implementation [here](http://https://github.com/percyliang/brown-cluster "here").
- We support the following set of languages for the postprocessing: *AFR*, *AMH*, *BUL*, *EUS*, *FIN*, *HIN*, *IND*, *JPN*, *LIT*, *PER*, *POR*, *TEL* and *TUR*.
- We support the UD POS tags.

 
