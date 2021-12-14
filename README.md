##  Unsupervised Cross-Lingual Part-of-Speech Tagging ##
##### Version: 1.5
<br/>

### Publications

- [Unsupervised Cross-Lingual Part-of-Speech Tagging for Truly Low-Resource Scenarios](https://www.aclweb.org/anthology/2020.emnlp-main.391.pdf "Unsupervised Cross-Lingual Part-of-Speech Tagging for Truly Low-Resource Scenarios")

---

### Requirements
- [GIZA++](http://www.statmt.org/moses/giza/GIZA++.html "GIZA++")
- [Python 3.x+](https://www.python.org/downloads/ "Python 3.x+")
- [PyTorch 1.1+]( https://pytorch.org/get-started/locally/ "PyTorch (>1.1)")
- [NumPy](https://numpy.org/ "NumPy")
- [Scikit-Learn](https://pypi.org/project/scikit-learn/ "Scikit-Learn")
- [MorphAGram](https://github.com/rnd2110/MorphAGram "MorphAGram") (Add a `MorphAGram` directory in the main directory of this repo.)

---

#### Producing the alignments

We use GIZA++ to train and produce word-level alignments between the target language and a source language for which POS annotations are available based on a parallel corpus that is white-space tokenized.

- Create a directory `alignments` that has the `GIZA++` and `mkcls` installation directories, in addition to the `run_gizapp.sh` and `giza-convert.py` scripts and a `workspace` directory to store the inputs and outputs.
- For the source language \<SL\> (ISO3 code), the target language \<TL\> (ISO3 code) and the dataset <D>, produce the following files:
    - the source-target GIZA++ input parallel file `\<SL\>-\<TL\>-<D>.parallel` (per line: <white_space_tokenized_source_sentence> ||| <white_space_tokenized_target_sentence>)
    - the GIZA++ input configuration file `\<SL\>-\<TL\>-<D>.giza.config`. Use the config file `data/gizapp-sample.config`, and replace 'ENG' by \<SL\>, 'AFR' by \<TL\> and 'bible' by <D>.
    - a key file of sentence IDs `\<SL\>-\<TL\>-<D>.keys`, one ID per line. The order of the IDs should correspond to the order of the sentences in `\<SL\>-\<TL\>-<D>.parallel`.
- Run the `run_gizapp.sh` script to train and produce the alignments from the source to the target with the three parameters \<SL\>, \<TL\> and <D>. This will create a new directory `workspace/\<SL\>-\<TL\>-<D>-gfiles` with the necessary GIZA++ output files.
- Run the `giza-convert.py` script to produce the forward alignments as follows: `python giza-convert.py workspace/\<SL\>-\<TL\>-<D>-gfiles/\<SL\>-\<TL\>.alignments N > workspace/\<SL\>-\<TL\>-<D>-giza.forward`
- Repeat the second and their steps while switching \<SL\> and \<TL\> in order to produce the backward alignments.
- Run the `giza-convert.py` script to produce the backward alignments as follows: `python giza-convert.py workspace/\<TL\>-\<SL\>-<D>-gfiles/\<TL\>-\<SL\>.alignments Y > workspace/\<SL\>-\<TL\>-<D>-giza.backward`

---

#### Tagging the Source

Use an off-the-shelf POS tagger to tag the source text. The output should be a tabular file that has one sentence per line, where the first tab has the sentence ID and the second tab has space-separated words and their POS tags as *word_POS*. 
Example:
`This_PRON is_AUX a_DET simple_ADJ example_NOUN ._PUNCT`

---

#### Annotation Projection

The  script `projection_handler.py` is responsible for projecting the tags from the source language onto the target one and coupling the token and type constraints. The script relies on the GIZA++ output along with the annotated source text.

##### Projection: projection_handler.py
##### Parameters
-  **key_path**: alignments/workspace/\<SL\>-\<TL\>-<D>.keys
-  **forward_alignment_path**: alignments/workspace/\<SL\>-\<TL\>-<D>-giza.forward
-  **forward_source_vocabulary_path**: alignments/workspace/\<SL\>-\<TL\>-<D>-gfiles/\<SL\>-\<TL\>.vcb
-  **forward_target_vocabulary_path**: alignments/workspace/\<SL\>-\<TL\>-<D>-gfiles/\<TL\>-\<SL\>.vcb
-  **forward_alignment_probability_path**: alignments/workspace/\<SL\>-\<TL\>-<D>-gfiles/g.t3.final
-  **backward_alignment_path**: alignments/workspace/\<SL\>-\<TL\>-<D>-giza.backward
-  **backward_source_vocabulary_path**: alignments/workspace/\<TL\>-\<SL\>-<D>-gfiles/\<SL\>-\<TL\>.vcb
-  **backward_target_vocabulary_path**: alignments/workspace/\<TL\>-\<SL\>-<D>-gfiles/\<TL\>-\<SL\>.vcb
-  **backward_alignment_probability_path**: alignments/workspace/\<TL\>-\<SL\>-<D>-gfiles/g.t3.final
-  **tagged_source_path**: the path of the tagged source text
-  **target_data_path**: the path of the target text, a tabular file that has one sentence per line, where the first tab has the sentence ID and the second tab has the white-space tokenized sentence.
-  **pos_output_path**: the path of the projected annotations
-  **pos_output_path**: the path of the probabilities of the projected annotations

In order to apply stem-based alignment and projection, run both alignment and projection in the stem space, and then replace the tagged stems by their corresponding words using the `map_word_stem.py` script.

##### Stem-based Processing: map_word_stem.py
##### Parameters
-  **source_data_path**: the path of the source text, a tabular file that has one sentence per line, where the first tab has the sentence ID and the second tab has the white-space tokenized sentence.
-  **tagged_stem_path**: the path of the stem-based projected annotations (as produced by `projection_handler.py`)
-  **pos_output_path**: the path of the output annotations in which the stems are replaced by their corresponding words

In order to apply morpheme-based alignment and projection, run both alignment and projection in the morpheme space, and then replace the tagged morphemes by their corresponding words and their tags by their representative tags using the `map_word_morpheme.py` script.

##### Morpheme-based Processing: map_word_morpheme.py
##### Parameters
-  **source_data_path**: the path of the source text
-  **tagged_morpheme_path**: the path of the morpheme-based projected annotations (as produced by `projection_handler.py`)
-  **prob_path**: the path of the probabilities of the morpheme-based projected annotations (as produced by `projection_handler.py`). This is not needed in the stem-based approach as the stem-based probabilities are the same as the word-based probabilities. 
-  **stem_path**: the path of the source text where the stems are marked by '+' e.g., 're+play+s' or '+make+'. This is needed in the STEM mechanism for the selection of the representative morpheme. If set to 'NA', the RANK mechanism is used instead.
-  **pos_output_path**: the path of the output annotations in which the morphemes are replaced by their corresponding words
-  **prob_output_path**: the path of the probabilities of the output annotations

The  script `multi_source_projection.py` is responsible for multi-source projection.

##### Multi-Source Projection: multi_source_projection.py
##### Parameters
-  **target_language**: the ISO3 code of the target language
-  **source_languages**: the comma-separated ISO3 codes of the source languages
-  **use_bea**: `T` = use Bayesian inference, weighted-maximum voting otherwise
-  **use_weights_with_bea**: `T` = use the weights in the initialization of Bayesian inference, unweighted otherwise
-  **use_weights_with_argmax**: `T` = use the weights in the argmax of Bayesian inference, unweighted otherwise
-  **scale_weights** `T` = scale the weights between 0 and 1, unscaled otherwise
-  **pos_path**: the path of the projected annotations (as produced by `projection_handler.py`). It is a wildcard path in which the source language is be replaced by '#SOURCE_LANG#'.
-  **prob_path**: the path of the probabilities of the projected annotations (as produced by `projection_handler.py`). It is a wildcard path in which the source language is be replaced by '#SOURCE_LANG#'.
-  **pos_output_path**: the path of the output multi-source projected annotations
-  **prob_output_path**: the path of the output probabilities of the multi-source projected annotations


The script `training_data_generator.py` is responsible for generating the training data given the projected annotations and their weights. It relies on the the outputs produced by either `projection_handler.py` or `multi_source_projection.py`. The produced POS training file has one sentence per line, where each word is represented as *word_POS*, and empty tags are marked as \*\*\*.  
Example:  
`Deur_ADP saam_*** te_PART werk_VERB ,_PUNCT kan_*** ons_PRON meer_DET bereik_VERB ._PUNCT`

##### Generation of Training Data: training_data_generator.py
##### Parameters
-  **pos_path**: the path of the projected annotations (as produced by `projection_handler.py` or `multi_source_projection.py`)
-  **prob_path**: the path of the probabilities of the projected annotations (as produced by `projection_handler.py` or `multi_source_projection.py`)
-  **training_path**: the path of the output training file

---

### Training and Testing the POS model

The script `tagger.py` is responsible for training and testing the neural POS tagger in one fell swoop. However, it should be straightforward to split the training and testing phases, if needed.

##### Training and Decoding: tagger.py
##### Parameters
-  **target_language**: the ISO3 code of the target language
-  **source_language**: the ISO3 code of the source language
- **data_path**: the path of the training and testing data (directory)
- **output_path**: the path of the final tagged output (directory)
- **model_path**: the path of the final model (directory)
- **training_data_set**: the name of the training dataset
- **test_data_sets**: the comma-separated name(s) of the test dataset(s) (or 'NA' for no testing)
- **training_size**: the number of words to train on, -1 = all
- **max_sentence_length**: the maximum sentence length (in words) to train on, -1 = all
- **min_density**: the percentage of partially tagged words to the number of words in a sentence to train on, -1 = all
- **use_word_embeddings**: whether to use randomly initialized word  embeddings (recommended)
- **use_length_affix_embeddings**: whether to use randomly initialized prefix/suffix (of lengths 1, 2, 3 and 4) embeddings (recommended).
- **use_segmentation_affix_embeddings**: whether to use randomly initialized MorphAGram prefix/suffix embeddings (requires MorphAGram and a segmentation model)
- **use_segmentation_complex_affix_embeddings**: whether to use randomly initialized MorphAGram complex prefix/suffix embeddings (requires MorphAGram and a segmentation model)
- **use_segmentation_stem_embeddings**: whether to use randomly initialized MorphAGram stem embeddings (requires MorphAGram and a segmentation model)
- **use_char_embeddings**: whether to use character embeddings
- **segmentation_grammar_output_path**: the path of the segmentation output based on the segmentation grammar needed for running MorphAGram, 'NA' otherwise
- **use_brown_clusters**: whether to use Brown clusters (recommended)
- **brown_cluster_path**: the path of Brown clusters
- **use_contextual_embeddings**: whether to use contextual embeddings (e.g., BERT or XLM) (recommended)
- **contextual_embeddings_dimensions**: the vector size for contextual embeddings
- **contextual_embedding_path**: the path of the precomputed contextual embeddings (see the description under the *Notes* section)
- **contextual_tokenization_path**: the path of the tokenization file, a tabular file of two columns: words and space-separated tokens
- **subword_combination_method**: how to combine the embeddings of subwords; the values are: *AVERAGE*, *FIRST*, *FIRST_LAST* and *LONGEST* (recommended: *FIRST_LAST*)
- **epochs**: number of epochs (recommended: 12)
- **learning_rate**: learning rate (recommended: 0.0001)
- **learning_decay_rate**: learning decay rate (recommended: 0.1)
- **dropout_rate**: dropout rate (recommended: 0.7)
- **fix_tags**: whether to match the output tags with the UD annotation guidelines for the underlying language (e.g., converting PRT to ADV in TUR)
- **run_postprocessing**: whether to force rule-based tagging for punctuation marks, symbols and numbers in the output (recommended)
- **overwrite_by_output**: whether to  use the tags in the test dataset(s) (e.g., when partially annotated) to overwrite the output of the system

##### Notes about Training the POS model
- The system assumes all the contextual embeddings are precomputed. However, it is straightforward to change this into runtime computations, if needed.
- In the embeddings file, each sentence should occupy n+2 lines. The first line contains the white-space tokenized text; the second line contains a vector of subwords or subword IDs, while the *(n+2)th* line contains the comma-separated vector of the *nth* token.  
Example:  
`Eta haur eçagut cedin Ioppe gucian eta sinhets ceçaten anhitzec Iauna baithan .`  
`['[CLS]', 25623, 56155, 28, 2968, 13306, 405, 3035, 28136, 7340, 2497, 69438, 522, 3811, 7831, 405, 2968, 510, 3616, 24374, 3240, 12044, 1946, 63670, 1121, 6, 5, '[SEP]']`  
`0.6276292204856873, -0.8384165167808533, 0.6102157235145569, -0.2547730505466461, -0.45138606429100037,.....`  
- The training fie should be named as *(target_language)-(source_languagonee)-(training_data_set).txt*, e.g., *EUS-ENG-TRAIN.txt*.
- The test file(s) hould be named as  *(target_language)-(source_language)-(test_data_set).txt*, e.g., *EUS-ENG-TEST.txt*.
- The test files should has the same format as the training one.
- We use the output of the Brown-Clustering implementation [here](http://https://github.com/percyliang/brown-cluster "here").
- The system supports the following set of languages for the postprocessing: *AFR*, *AMH*, *BUL*, *EUS*, *FIN*, *HIN*, *IND*, *KAT*, *KAZ*, *JPN*, *LIT*, *PER*, *POR*, *TEL* and *TUR*.
- The booleans are expressed as 'Yes', 'True', 'T', 'Y' or '1' for the TRUE value.

The script `multi_source_decoding.py` is responsible for combining decoded outputs that correspond to single-source POS models.

##### Multi_source Decoding: multi_source_decoding.py
##### Parameters
-  **target_language**: the ISO3 code of the target language
-  **source_languages**: the comma-separated ISO3 codes of the source languages
-  **use_bea**: `T` = use Bayesian inference, weighted-maximum voting otherwise
-  **use_inference_weights**: `T` = use inference weights, alignment probabilities otherwise
-  **use_weights_with_bea**: `T` = use  the weights in the initialization of Bayesian inference, unweighted otherwise
-  **use_weights_with_argmax**: `T` = use the weights in the argmax of Bayesian inference, unweighted otherwise
-  **scale_weights** `T` = scale the weights between 0 and 1, unscaled otherwise
-  **pos_path**: the path of the decoded POS annotations (as produced by `tagger.py` - only used with inference-based weights). It is a wildcard path in which the source language is be replaced by '#SOURCE_LANG#'.
-  **prob_path**: the path of the probabilities of the decoded POS annotations (as produced by `tagger.py` - only used with inference-based weights). It is a wildcard path in which the source language is be replaced by '#SOURCE_LANG#'.
-  **training_path**: the path of the training files (as produced by `training_data_generator.py` - only used for statistics). It is a wildcard path in which the source language is be replaced by '#SOURCE_LANG#'.
-  **forward_alignment_path**: alignments/workspace/\<SL\>-\<TL\>-<D>-giza.forward - only used with alignment-based weights. It is a wildcard path in which the source language is be replaced by '#SOURCE_LANG#'.
-  **backward_alignment_path**: alignments/workspace/\<SL\>-\<TL\>-<D>-giza.backward - only used with alignment-based weights. It is a wildcard path in which the source language is be replaced by '#SOURCE_LANG#'.
-  **gold_sets**: the name of the gold/test dataset (for evaluation purpose)
-  **pos_output_file**: the path of the gold file (for evaluation purpose)
-  **pos_output_file**: the path of the output combined POS tags
-  **eval_output_file**: the path of the output evaluation script (for evaluation purpose)

### Acknowledgement
    
This research is based upon work supported by the Intelligence Advanced Research Projects Activity (IARPA), (contract FA8650-17-C-9117). The views and conclusions herein are those of the authors and should not be interpreted as necessarily representing official policies, ex-pressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes
notwithstanding any copy-right annotation therein.
