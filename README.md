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

####Main Script: tagger.py

##### Parameters
-  **target_language**: The ISO3 code of the target language
-  **source_language**: The ISO3 code of the target language
- **data_path**: The path of the training and testing data (directory)
- **output_path**: The path of the final tagged output (directory)
- **model_path**: The path of the final model (directory)
- **training_data_set**: The name of the training dataset
- **test_data_sets**: The name(s) of the test dataset(s), comma-separated
- **training_size**: the number of words to train one, -1 = all
- **max_sentence_length**: the maximum sentence length (in words) to train one, -1 = all
- **min_density**: the percentage of partially tagged words to the number of words in a sentence to train on, -1 = all
- **use_word_embeddings**: boolean: whether to use randomly initializedword  embeddings.
- **use_affix_embeddings**: whether to use randomly initialized prefix/suffix (of lengths 1, 2, 3 and 4) embeddings.
- **use_char_embeddings**: whether to use character embeddings
- **use_brown_clusters**: whether to use brown clusters
- **brown_cluster_path**: the path of brown clusters
- **use_contextual_embeddings**: whether to use contextual embeddings (e.g., BERT or XLM)
- **contextual_embeddings_dimensions**: size of the embedding dimensions.
- **contextual_embedding_path**: the path of the contextual embedding vevtors.  Each sentence should occypy n+1 lines. The first line contains the space-separated tokenized text, while the nth line contains the comma separated embeddings values of the (n-1)th token (e.g., 0.2, 0.3, -.0.6....).
- **contextual_tokenization_path**: the path olf the tokenization file, a tabular file of two columns
- **subword_combination_method**: how to combine the embeddings of subwords; the values are: AVERAGE, FIRST, FIRST_LAST and LONGEST.
- **epochs**: number of epochs (recommended: 12)
- **learning_rate**: learnign rate )recommended: 0.0001)
- **learning_decay_rate**: learning decay rate (recommended: 0.1)
- **dropout_rate**: dropout rate (recommended between 0.7)
- **fix_tags**: whether to match the output tags with the UD annotation guidlunes for tge underlying languages (e.g., converting PRT to ADV in TUR)
- **run_postprocessing**: whether to force the tagging of punctuation marks, symbols and numbers in the output
- **overwrite_by_output**: whether to  use the tags in the test dataset(s) (e.g., when partially annotated) to overwrite the system output

#### Notes
- The system assumes all the contextual embeddings are precomputed. However, it is straightforward to change this into runtime computations, if needed.
- The training dataset should be named as *(target_language)-(source_language)-POSUD-(training_data_set).txt*, e.g., *EUS-ENG-POSUD-TRAIN.txt*.
- The test dataset should be named as  *(target_language)-(source_language)-POSUD-(test_data_sets).txt*, e.g., *EUS-ENG-POSUD-TEST.txt*.
- The training and testing datasets should have ne sentcne per line, where each words is represented as **word_POS** and embetty tags are marked as \*\*\*.
Example:  `Deur_ADP saam_ADV te_PART werk_VERB ,_PUNCT kan_AUX ons_PRON meer_DET bereik_VERB ._PUNCT`
- We use the output of the Brown-Clustering implementation [here](http://https://github.com/percyliang/brown-cluster "here").

 
