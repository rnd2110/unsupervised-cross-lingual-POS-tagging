#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random

import torch
import numpy as np

from data_reader import *
from brown_clustering_processing import *
from contextual_embedding_map import *

#Parameters
EMBEDDING_DIM = 64
LSTM_HIDDEN_DIM = 128
EPOCHS = 12
BATCH_SIZE = 2
LEARNING_RATE = 0.0001
LEARNING_DECAY_RATE = 0.1
DROPOUT_RATE = 0.7
WEIGHT_DECAY = 1e-8
BROWN_SIZE = 16

TARGET_LANGUAGE = ''
SOURCE_LANGUAGE = ''
DATA_PATH = ''
OUTPUT_PATH = ''
MODEL_PATH = ''
TRAINING_DATA_SET = ''
TEST_DATA_SETS = ''

TRAINING_SIZE = -1
MAX_SENTENCE_LENGTH = -1
MIN_DENSITY = 0.49999

USE_WORD_EMBEDDINGS = True
USE_AFFIX_EMBEDDINGS = True
USE_CHAR_EMBEDDINGS = True
USE_BROWN_CLUSTERS = True
BROWN_CLUSTER_PATH = ''
USE_CONTEXTUAL_EMBEDDINGS = True
CONTEXTUAL_EMBEDDING_DIM = 1024
CONTEXTUAL_EMBEDDING_PATH = ''
CONTEXTUAL_TOKENIZATION_PATH = ''
SUBWORD_COMBINATION_METHOD = 'AVERAGE', #FIRST, AVERAGE, FIRST_LAST, LONGEST

FIX_TAGS = True
RUN_POSTPROCESSING = False
OVERWRITE_BY_OUTPUT = False

seed_value=1
#os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

torch.set_num_threads(1)

def generate_random_vector(seed, min_value, max_value, dim1, dim2):
    np.random.seed(seed)
    rand_array = np.random.uniform(min_value,max_value,[dim1, dim2])
    return rand_array

def main():

    #### Read the arguments
    
    parser = argparse.ArgumentParser(description='POS Tagger')
    parser.add_argument('target_language')
    parser.add_argument('source_language')
    parser.add_argument('data_path')
    parser.add_argument('output_path')
    parser.add_argument('model_path')
    parser.add_argument('training_data_set')
    parser.add_argument('test_data_sets')
    parser.add_argument('training_size')
    parser.add_argument('max_sentence_length')
    parser.add_argument('min_density')
    parser.add_argument('use_word_embeddings')
    parser.add_argument('use_affix_embeddings')
    parser.add_argument('use_char_embeddings')
    parser.add_argument('use_brown_clusters')
    parser.add_argument('brown_cluster_path')
    parser.add_argument('use_contextual_embeddings')
    parser.add_argument('contextual_embedding_dim')
    parser.add_argument('contextual_embedding_path')
    parser.add_argument('contextual_tokenization_path')
    parser.add_argument('subword_combination_method')
    parser.add_argument('epochs')
    parser.add_argument('learning_rate')
    parser.add_argument('learning_decay_rate')
    parser.add_argument('dropout_rate')
    parser.add_argument('fix_tags')
    parser.add_argument('run_postprocessing')
    parser.add_argument('overwrite_by_output')
    
    args = parser.parse_args()
    
    params = {}

    params['TARGET_LANGUAGE'] = TARGET_LANGUAGE
    params['SOURCE_LANGUAGE'] = SOURCE_LANGUAGE
    params['DATA_PATH'] = DATA_PATH
    params['OUTPUT_PATH'] = OUTPUT_PATH
    params['MODEL_PATH'] = MODEL_PATH
    params['TRAINING_DATA_SET'] = TRAINING_DATA_SET
    params['TEST_DATA_SETS'] = TEST_DATA_SETS
    params['TRAINING_SIZE'] = TRAINING_SIZE
    params['MAX_SENTENCE_LENGTH'] = MAX_SENTENCE_LENGTH
    params['MIN_DENSITY'] = MIN_DENSITY
    params['USE_WORD_EMBEDDINGS'] = USE_WORD_EMBEDDINGS
    params['USE_AFFIX_EMBEDDINGS'] = USE_AFFIX_EMBEDDINGS
    params['USE_CHAR_EMBEDDINGS'] = USE_CHAR_EMBEDDINGS
    params['USE_BROWN_CLUSTERS'] = USE_BROWN_CLUSTERS
    params['BROWN_CLUSTER_PATH'] = BROWN_CLUSTER_PATH
    params['USE_CONTEXTUAL_EMBEDDINGS'] = USE_CONTEXTUAL_EMBEDDINGS
    params['CONTEXTUAL_EMBEDDING_DIM'] = CONTEXTUAL_EMBEDDING_DIM
    params['CONTEXTUAL_EMBEDDING_PATH'] = CONTEXTUAL_EMBEDDING_PATH
    params['CONTEXTUAL_TOKENIZATION_PATH'] = CONTEXTUAL_TOKENIZATION_PATH
    params['SUBWORD_COMBINATION_METHOD'] = SUBWORD_COMBINATION_METHOD 
    params['EPOCHS'] = EPOCHS
    params['LEARNING_RATE'] = LEARNING_RATE
    params['LEARNING_DECAY_RATE'] = LEARNING_DECAY_RATE
    params['DROPOUT_RATE'] = DROPOUT_RATE
    params['FIX_TAGS'] = FIX_TAGS
    params['RUN_POSTPROCESSING'] = RUN_POSTPROCESSING
    params['OVERWRITE_BY_OUTPUT'] = OVERWRITE_BY_OUTPUT

    if args.target_language is not None:
        params['TARGET_LANGUAGE'] = args.target_language
    if args.source_language is not None:
        params['SOURCE_LANGUAGE'] = args.source_language
    if args.data_path is not None:
        params['DATA_PATH'] = args.data_path
    if args.output_path is not None:
        params['OUTPUT_PATH'] = args.output_path
    if args.model_path is not None:
        params['MODEL_PATH'] = args.model_path
    if args.training_data_set is not None:
        params['TRAINING_DATA_SET'] = args.training_data_set
    if args.test_data_sets is not None:
        params['TEST_DATA_SETS'] = args.test_data_sets
    if args.training_size is not None:
        params['TRAINING_SIZE'] = int(args.training_size)
    if args.max_sentence_length is not None:
        params['MAX_SENTENCE_LENGTH'] = int(args.max_sentence_length)
    if args.min_density is not None:
        params['MIN_DENSITY'] = float(args.min_density)
    if args.use_word_embeddings is not None:
        params['USE_WORD_EMBEDDINGS'] = str2bool(args.use_word_embeddings)
    if args.use_affix_embeddings is not None:
        params['USE_AFFIX_EMBEDDINGS'] = str2bool(args.use_affix_embeddings)
    if args.use_char_embeddings is not None:
        params['USE_CHAR_EMBEDDINGS'] = str2bool(args.use_char_embeddings)
    if args.use_brown_clusters is not None:
        params['USE_BROWN_CLUSTERS'] = str2bool(args.use_brown_clusters)
    if args.brown_cluster_path is not None:
        params['BROWN_CLUSTER_PATH'] = args.brown_cluster_path
    if args.use_contextual_embeddings is not None:
        params['USE_CONTEXTUAL_EMBEDDINGS'] = str2bool(args.use_contextual_embeddings)
    if parser.add_argument('contextual_embedding_dim'):
        params['CONTEXTUAL_EMBEDDING_DIM'] = int(args.contextual_embedding_dim)
    if args.contextual_embedding_path is not None:
        params['CONTEXTUAL_EMBEDDING_PATH'] = args.contextual_embedding_path
    if args.contextual_tokenization_path is not None:
        params['CONTEXTUAL_TOKENIZATION_PATH'] = args.contextual_tokenization_path
    if args.subword_combination_method is not None:
        params['SUBWORD_COMBINATION_METHOD'] = args.subword_combination_method
    if args.epochs is not None:
        params['EPOCHS'] = int(args.epochs)
    if args.learning_rate is not None:
        params['LEARNING_RATE'] = float(args.learning_rate)
    if args.learning_decay_rate is not None:
        params['LEARNING_DECAY_RATE'] = float(args.learning_decay_rate)
    if args.dropout_rate is not None:
        params['DROPOUT_RATE'] = float(args.dropout_rate)
    if args.fix_tags is not None:
        params['FIX_TAGS'] = str2bool(args.fix_tags)
    if args.run_postprocessing is not None:
        params['RUN_POSTPROCESSING'] = str2bool(args.run_postprocessing)
    if args.overwrite_by_output is not None:
        params['OVERWRITE_BY_OUTPUT'] = str2bool(args.overwrite_by_output)

    print("Reading data...")
    train, tune = read_data(True, params['TARGET_LANGUAGE'], params['SOURCE_LANGUAGE'], params['DATA_PATH'], params['TRAINING_DATA_SET'], params['TRAINING_SIZE'], params['MAX_SENTENCE_LENGTH'],  params['MIN_DENSITY'], True)
    tests = []
    test_data_set_map = {}
    test_data_set_list = params['TEST_DATA_SETS'].split(',')
    for test_data_set in test_data_set_list:
        test,_ = read_data(False, params['TARGET_LANGUAGE'], params['SOURCE_LANGUAGE'], params['DATA_PATH'], test_data_set, -1, -1, params['MIN_DENSITY'], False)
        tests.append(test)
        test_data_set_map[test_data_set] = test
    vocab_processor = Vocab(train, tune, tests)
    print("Training size: ", len(train))

    # Create output and model directories if needed
    if not os.path.exists(params['OUTPUT_PATH']):
        os.makedirs(params['OUTPUT_PATH'])
    if not os.path.exists(params['MODEL_PATH']):
        os.makedirs(params['MODEL_PATH'])
        
    brown_processor = None
    if params['USE_BROWN_CLUSTERS']:
        print("Loading Brown clusters...")
        brown_processor = BROWN_Processor(params['BROWN_CLUSTER_PATH'], BROWN_SIZE)

    contextual_embedding_map = None
    if params['USE_CONTEXTUAL_EMBEDDINGS']:
        print("Loading contextual embeddings...")
        sentences = []
        all_data = train + tune
        for test_data_set in test_data_set_map:
            all_data = all_data + test_data_set_map[test_data_set]
        for n, (tokens, tags) in enumerate(all_data):        
            sentence = ' '.join([simplify_token(word) for word in tokens][1:-1])
            sentences.append(sentence)
        contextual_embedding_map = Contextual_Embedding_Map(params['CONTEXTUAL_EMBEDDING_PATH'], params['CONTEXTUAL_TOKENIZATION_PATH'], params['SUBWORD_COMBINATION_METHOD'], sentences)
        contextual_embedding_map.load_embeddings()

    #### Model creation
    WORD_LSTM_DIM = EMBEDDING_DIM if params['USE_WORD_EMBEDDINGS'] else 0
    AFFIX_LSTM_DIM = 8 * EMBEDDING_DIM if params['USE_AFFIX_EMBEDDINGS'] else 0
    CHAR_LSTM_DIM = (EMBEDDING_DIM * vocab_processor.max_token_length()) if params['USE_CHAR_EMBEDDINGS'] else 0
    BROWN_LSTM_DIM = 10 * (BROWN_SIZE +1 if params['USE_BROWN_CLUSTERS'] else 0)
    CONTEXTUAL_LSTM_DIM = params['CONTEXTUAL_EMBEDDING_DIM']  if params['USE_CONTEXTUAL_EMBEDDINGS'] else 0
    dimensions = WORD_LSTM_DIM + AFFIX_LSTM_DIM + CHAR_LSTM_DIM + BROWN_LSTM_DIM + CONTEXTUAL_LSTM_DIM
    print('Input Dimensions: ', dimensions)
    model = TaggerModel(params, vocab_processor, dimensions)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['LEARNING_RATE'], weight_decay=WEIGHT_DECAY)

    scheduler = None
    if params['LEARNING_DECAY_RATE'] > 0.0000001:
        rescale_lr = lambda epoch: 1 / (1 + params['LEARNING_DECAY_RATE'] * epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rescale_lr)

    #### Group together framework-specific variables needed when iterating over the data.
    expressions = (model, optimizer)
    
    #### Main training itertion
    max_tune_acc = 0
    for epoch in range(params['EPOCHS']):
        random.shuffle(train)

        print("Learning rate: ", epoch, optimizer.param_groups[0]["lr"]);

        model.train() 
        model.zero_grad()

        #### Training pass
        train_loss, train_acc1, _, _ = do_pass(train, expressions, True, vocab_processor, brown_processor, contextual_embedding_map, params, False, None, None)
        ####
        model.eval()

        #### Tuning pass
        _, train_acc2, _, _ = do_pass(train, expressions, False, vocab_processor, brown_processor, contextual_embedding_map, params, False, None, None)
        _, tune_acc, _, _ = do_pass(tune, expressions, False, vocab_processor, brown_processor, contextual_embedding_map, params, False, None, None)

        print("@@@ {} loss: {:.5f} train-acc1: {:.3f} train-acc2: {:.5f} tune-acc: {:.5f}".format(epoch, train_loss, train_acc1, train_acc2, tune_acc))

        #### Testing pass
        for test_data_set in test_data_set_map:
            writer = open(params['OUTPUT_PATH']+'/'+params['TARGET_LANGUAGE']+'-'+params['SOURCE_LANGUAGE']+'-'+test_data_set+'-'+str(epoch)+'.txt', 'w')
            _, test_acc, test_acc_petrov, info = do_pass(test_data_set_map[test_data_set], expressions, False, vocab_processor, brown_processor, contextual_embedding_map, params, params['FIX_TAGS'], test_data_set, writer)
            writer.close()
            print("### " + params['TARGET_LANGUAGE']+'-'+params['SOURCE_LANGUAGE']+'-'+test_data_set+'-'+str(epoch) + ' ' + str(info) + " test-acc-petrov: {:.5f}".format(test_acc_petrov) + " test-acc: {:.5f}".format(test_acc))

        # Save model
        if epoch == params['EPOCHS']-1:
            torch.save(model.state_dict(), params['MODEL_PATH']+'/'+params['TARGET_LANGUAGE']+'-'+params['SOURCE_LANGUAGE']+'-'+params['TRAINING_DATA_SET']+'-'+str(epoch)+'.model')

        #### Update learning rate
        if scheduler is not None:
            scheduler.step()

    ### Load model
    #model.load_state_dict(torch.load(params['MODEL_PATH']), strict=False)

def weights_init(m):
    classname = m.__class__.__name__ # python trick that will look for the type of connection in the object "m" (convolution or full connection)
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size()) #?? list containing the shape of the weights in the object "m"
        fan_in = weight_shape[1] # dim1
        fan_out = weight_shape[0] # dim0
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros
            
class TaggerModel(torch.nn.Module):
    #### In the constructor we define objects that will do each of the computations.

    def __init__(self, params, vocab_processor, dimensions):
        super().__init__()
        
        ### Create embeddings
        word_embedding_tensor = torch.FloatTensor(generate_random_vector(1, -1, 1, vocab_processor.num_tokens(), EMBEDDING_DIM))
        self.word_embedding = torch.nn.Embedding.from_pretrained(word_embedding_tensor, freeze=False)

        prefix1_embedding_tensor = torch.FloatTensor(generate_random_vector(2, -1, 1, vocab_processor.num_prefixes1(), EMBEDDING_DIM))
        self.prefix1_embedding = torch.nn.Embedding.from_pretrained(prefix1_embedding_tensor, freeze=False)

        prefix2_embedding_tensor = torch.FloatTensor(generate_random_vector(3, -1, 1, vocab_processor.num_prefixes2(), EMBEDDING_DIM))
        self.prefix2_embedding = torch.nn.Embedding.from_pretrained(prefix2_embedding_tensor, freeze=False)

        prefix3_embedding_tensor = torch.FloatTensor(generate_random_vector(4, -1, 1, vocab_processor.num_prefixes3(), EMBEDDING_DIM))
        self.prefix3_embedding = torch.nn.Embedding.from_pretrained(prefix3_embedding_tensor, freeze=False)

        prefix4_embedding_tensor = torch.FloatTensor(generate_random_vector(5, -1, 1, vocab_processor.num_prefixes4(), EMBEDDING_DIM))
        self.prefix4_embedding = torch.nn.Embedding.from_pretrained(prefix4_embedding_tensor, freeze=False)

        suffix1_embedding_tensor = torch.FloatTensor(generate_random_vector(6, -1, 1, vocab_processor.num_suffixes1(), EMBEDDING_DIM))
        self.suffix1_embedding = torch.nn.Embedding.from_pretrained(suffix1_embedding_tensor, freeze=False)

        suffix2_embedding_tensor = torch.FloatTensor(generate_random_vector(7, -1, 1, vocab_processor.num_suffixes2(), EMBEDDING_DIM))
        self.suffix2_embedding = torch.nn.Embedding.from_pretrained(suffix2_embedding_tensor, freeze=False)

        suffix3_embedding_tensor = torch.FloatTensor(generate_random_vector(8, -1, 1, vocab_processor.num_suffixes3(), EMBEDDING_DIM))
        self.suffix3_embedding = torch.nn.Embedding.from_pretrained(suffix3_embedding_tensor, freeze=False)

        suffix4_embedding_tensor = torch.FloatTensor(generate_random_vector(9, -1, 1, vocab_processor.num_suffixes4(), EMBEDDING_DIM))
        self.suffix4_embedding = torch.nn.Embedding.from_pretrained(suffix4_embedding_tensor, freeze=False)

        char_embedding_tensor = torch.FloatTensor(generate_random_vector(10, -1, 1, vocab_processor.num_characters(), EMBEDDING_DIM))
        self.chars_embedding = torch.nn.Embedding.from_pretrained(char_embedding_tensor, freeze=False)
        
        ### Create input dropout parameter
        self.dropout = torch.nn.Dropout(params['DROPOUT_RATE'])
                                                     
        ### Create LSTM parameters
        self.lstm = torch.nn.LSTM(dimensions, LSTM_HIDDEN_DIM, num_layers=1, batch_first=True, bidirectional=True)

        ### Initilizing the weights of the model with random weights
        self.apply(weights_init)
        
        # Create output dropout parameter
        self.lstm_output_dropout = torch.nn.Dropout(params['DROPOUT_RATE'])
                                                     
        # Create final matrix multiply parameters
        self.hidden_to_tag = torch.nn.Linear(LSTM_HIDDEN_DIM * 2, vocab_processor.num_tags())

        #self.crf = CRF(vocab_processor.num_tags())
        
    def forward(self, words, prefixes1, prefixes2, prefixes3, prefixes4, suffixes1, suffixes2, suffixes3, suffixes4, chars_list, brown_vectors, contextual_vectors, labels, lengths, cur_batch_size, vocab_processor, params):

        max_length = words.size(1)

        ### Look up word vectors
        word_vectors = self.word_embedding(words)
        prefix1_vectors = self.prefix1_embedding(prefixes1)
        prefix2_vectors = self.prefix2_embedding(prefixes2)
        prefix3_vectors = self.prefix3_embedding(prefixes3)
        prefix4_vectors = self.prefix4_embedding(prefixes4)
        suffix1_vectors = self.suffix1_embedding(suffixes1)
        suffix2_vectors = self.suffix2_embedding(suffixes2)
        suffix3_vectors = self.suffix3_embedding(suffixes3)
        suffix4_vectors = self.suffix4_embedding(suffixes4)
        chars_vectors = []
        for chars in chars_list:
            chars_vectors.append(self.suffix4_embedding(chars))

        ### Apply dropout
        if params['USE_WORD_EMBEDDINGS']:
            word_vectors = self.dropout(word_vectors)
        if params['USE_AFFIX_EMBEDDINGS']:
            prefix1_vectors = self.dropout(prefix1_vectors)
            prefix2_vectors = self.dropout(prefix2_vectors)
            prefix3_vectors = self.dropout(prefix3_vectors)
            prefix4_vectors = self.dropout(prefix4_vectors)
            suffix1_vectors = self.dropout(suffix1_vectors)
            suffix2_vectors = self.dropout(suffix2_vectors)
            suffix3_vectors = self.dropout(suffix3_vectors)
            suffix4_vectors = self.dropout(suffix4_vectors)
        if params['USE_CHAR_EMBEDDINGS']:
            for chars_vector in chars_vectors:
                chars_vector = self.dropout(chars_vector)
        #if params['USE_BROWN_CLUSTERS']:
            #brown_vectors = self.dropout(brown_vectors)
        if params['USE_CONTEXTUAL_EMBEDDINGS']:
            contextual_vectors = self.dropout(contextual_vectors)

        ### Concatenation of word-repreentation variables
        conc_tensor = None
        if params['USE_WORD_EMBEDDINGS']:
            conc_tensor = torch.cat((conc_tensor, word_vectors), -1) if conc_tensor is not None else word_vectors
        if params['USE_AFFIX_EMBEDDINGS']:
            conc_tensor = torch.cat((conc_tensor, prefix1_vectors, prefix2_vectors, prefix3_vectors, prefix4_vectors, suffix1_vectors, suffix2_vectors, suffix3_vectors, suffix4_vectors), -1) if conc_tensor is not None else torch.cat((prefix1_vectors, prefix2_vectors, prefix3_vectors, prefix4_vectors, suffix1_vectors, suffix2_vectors, suffix3_vectors, suffix4_vectors), -1)
        if params['USE_CHAR_EMBEDDINGS']:
            conc_tensor = torch.cat((conc_tensor, torch.cat(chars_vectors, -1)), -1) if conc_tensor is not None else torch.cat(chars_vectors, -1)
        if params['USE_BROWN_CLUSTERS']:
            conc_tensor = torch.cat((conc_tensor, brown_vectors), -1) if conc_tensor is not None else brown_vectors
        if params['USE_CONTEXTUAL_EMBEDDINGS']:
            conc_tensor = torch.cat((conc_tensor, contextual_vectors), -1) if conc_tensor is not None else contextual_vectors

        # Run LSTM
        conc_tensor = torch.nn.utils.rnn.pack_padded_sequence(conc_tensor, lengths, True)
                                                    
        lstm_out, _ = self.lstm(conc_tensor, None)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=max_length)
        ### Apply dropout
        lstm_out_dropped = self.lstm_output_dropout(lstm_out)
        output_scores = self.hidden_to_tag(lstm_out_dropped)

        # Calculate loss and predictions
        output_scores = output_scores.view(cur_batch_size * max_length, -1)
        flat_labels = labels.view(cur_batch_size * max_length)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        loss = loss_function(output_scores, flat_labels)

        label_array = []
        for i in range(cur_batch_size):
            label_array.append(labels[i].detach().numpy())
            
        predicted_tags = []

        words_array = words.detach().numpy()

        for i in range(len(output_scores)):
            output_score = output_scores[i].detach().numpy();
            label = label_array[i//max_length][i%max_length]
            argmax_pred = 0
            argmax_pred_tags = []
            if params['OVERWRITE_BY_OUTPUT'] and label != vocab_processor.blank_tag_id():
                argmax_pred = label
            else:
                argmax_pred = np.argmax(output_score)
                if argmax_pred == vocab_processor.blank_tag_id():
                    argmax_pred = np.argsort(output_score, axis=0)[len(output_score)-2]
            argmax_pred_tags.append(argmax_pred)                                    
            predicted_tags.append(argmax_pred_tags)
        flat = [item for sublist in predicted_tags for item in sublist]
        predicted_tags = torch.FloatTensor(flat)

        predicted_tags = predicted_tags.view(cur_batch_size, max_length)
        return loss, predicted_tags

#### Inference
def do_pass(data, expressions, train, vocab_processor, brown_processor, contextual_embedding_map, params, fix_tags, test_data_set, writer):

    model, optimizer = expressions

    match = 0
    match_petrov = 0
    total = 0

    IV_match = 0
    IV_total = 0

    OOV_match = 0
    OOV_total = 0

    pos_gold_map = {}
    pos_predicted_map = {}
    pos_match_map = {}

    loss = 0

    # Loop over batches
    for start in range(0, len(data), BATCH_SIZE):
        #### Form the batch and order it based on length (important for efficient processing in PyTorch).
        batch = data[start : start + BATCH_SIZE]
        batch.sort(key = lambda x: -len(x[0]))

        ####
        # Prepare inputs
        #### Prepare input arrays, using .long() to cast the type from Tensor to LongTensor.
        cur_batch_size = len(batch)
        max_length = len(batch[0][0])
        lengths = [len(v[0]) for v in batch]
        input_word_array = torch.zeros((cur_batch_size, max_length)).long()
        input_prefix1_array = torch.zeros((cur_batch_size, max_length)).long()
        input_prefix2_array = torch.zeros((cur_batch_size, max_length)).long()
        input_prefix3_array = torch.zeros((cur_batch_size, max_length)).long()
        input_prefix4_array = torch.zeros((cur_batch_size, max_length)).long()
        input_suffix1_array = torch.zeros((cur_batch_size, max_length)).long()
        input_suffix2_array = torch.zeros((cur_batch_size, max_length)).long()
        input_suffix3_array = torch.zeros((cur_batch_size, max_length)).long()
        input_suffix4_array = torch.zeros((cur_batch_size, max_length)).long()
        input_char_arrays = []
        for i in range(vocab_processor.max_token_length()):
            input_char_arrays.append(torch.zeros((cur_batch_size, max_length)).long())
        input_brown_array = []
        input_contextual_array = []
        
        output_array = torch.zeros((cur_batch_size, max_length)).long()
                                                     
        #### Convert tokens and tags from strings to numbers using the indices.
        for n, (tokens, tags) in enumerate(batch):
            token_ids = [vocab_processor.token2id(simplify_token(t)) for t in tokens]
            tag_ids = [vocab_processor.tag2id(t) for t in tags]

            prefix1_list = []
            prefix2_list = []
            prefix3_list = []
            prefix4_list = []
            suffix1_list = []
            suffix2_list = []
            suffix3_list = []
            suffix4_list = []
            char_list = []

            for word in tokens:
                word, prefix1, prefix2, prefix3, prefix4, suffix1, suffix2, suffix3, suffix4, chars = vocab_processor.get_features(word)
                prefix1_list.append(prefix1)
                prefix2_list.append(prefix2)
                prefix3_list.append(prefix3)
                prefix4_list.append(prefix4)
                suffix1_list.append(suffix1)
                suffix2_list.append(suffix2)
                suffix3_list.append(suffix3)
                suffix4_list.append(suffix4)
                char_list.append(chars)

            prefix1_ids = [vocab_processor.prefix12id(prefix_feat) for prefix_feat in prefix1_list]
            prefix2_ids = [vocab_processor.prefix22id(prefix_feat) for prefix_feat in prefix2_list]
            prefix3_ids = [vocab_processor.prefix32id(prefix_feat) for prefix_feat in prefix3_list]
            prefix4_ids = [vocab_processor.prefix42id(prefix_feat) for prefix_feat in prefix4_list]
            suffix1_ids = [vocab_processor.suffix12id(suffix_feat) for suffix_feat in suffix1_list]
            suffix2_ids = [vocab_processor.suffix22id(suffix_feat) for suffix_feat in suffix2_list]
            suffix3_ids = [vocab_processor.suffix32id(suffix_feat) for suffix_feat in suffix3_list]
            suffix4_ids = [vocab_processor.suffix42id(suffix_feat) for suffix_feat in suffix4_list]
            char_ids = []
            for i in range(vocab_processor.max_token_length()):
                char_ids.append([vocab_processor.character2id(chars[i]) for chars in char_list])

            #### Fill the arrays, leaving the remaining values as zero (our padding value).
            input_word_array[n, :len(tokens)] = torch.LongTensor(token_ids)
            input_prefix1_array[n, :len(tokens)] = torch.LongTensor(prefix1_ids)
            input_prefix2_array[n, :len(tokens)] = torch.LongTensor(prefix2_ids)
            input_prefix3_array[n, :len(tokens)] = torch.LongTensor(prefix3_ids)
            input_prefix4_array[n, :len(tokens)] = torch.LongTensor(prefix4_ids)
            input_suffix1_array[n, :len(tokens)] = torch.LongTensor(suffix1_ids)
            input_suffix2_array[n, :len(tokens)] = torch.LongTensor(suffix2_ids)
            input_suffix3_array[n, :len(tokens)] = torch.LongTensor(suffix3_ids)
            input_suffix4_array[n, :len(tokens)] = torch.LongTensor(suffix4_ids)

            for i in range(vocab_processor.max_token_length()):
                input_char_arrays[i][n, :len(tokens)] = torch.LongTensor(char_ids[i])

            if params['USE_BROWN_CLUSTERS']:
                brown_clusters = [torch.FloatTensor(brown_processor.get_brown_cluster(simplify_token(word))) for word in tokens]
                for i in range(max_length-len(tokens)):
                    brown_clusters.append(torch.zeros(10*(BROWN_SIZE+1)).float())
                    x = torch.stack(brown_clusters)
                input_brown_array.append(torch.stack(brown_clusters))

            if params['USE_CONTEXTUAL_EMBEDDINGS']:
                sentence = ' '.join([simplify_token(word) for word in tokens][1:-1])
                contextual_clusters = [torch.FloatTensor(contextual_word_vector) for contextual_word_vector in contextual_embedding_map.contextual_embeddings[sentence]]
                contextual_clusters = [torch.FloatTensor([-1]*params['CONTEXTUAL_EMBEDDING_DIM'])] + contextual_clusters + [torch.FloatTensor([1]*params['CONTEXTUAL_EMBEDDING_DIM'])]
                for i in range(max_length-len(tokens)):
                    contextual_clusters.append(torch.zeros((params['CONTEXTUAL_EMBEDDING_DIM'])).float())
                input_contextual_array.append(torch.stack(contextual_clusters))
                           
            output_array[n, :len(tags)] = torch.LongTensor(tag_ids)

        brown_vectors = torch.stack(input_brown_array) if params['USE_BROWN_CLUSTERS'] else None
        contextual_vectors = torch.stack(input_contextual_array) if params['USE_CONTEXTUAL_EMBEDDINGS'] else None
        
        #### Construct computation
        batch_loss, output = model(input_word_array, input_prefix1_array, input_prefix2_array, input_prefix3_array, input_prefix4_array, input_suffix1_array, input_suffix2_array, input_suffix3_array, input_suffix4_array, input_char_arrays, brown_vectors, contextual_vectors, output_array, lengths, cur_batch_size, vocab_processor, params)

        #### Run computations
        if train:
            batch_loss.backward()
            optimizer.step()
            model.zero_grad()
            loss += batch_loss.item()

        predictions = output.cpu().data.numpy()

        #### Postprocess the predictions if required
        if params['RUN_POSTPROCESSING']:
            postprocessed_predictions = []
            for (tokens, _), pred in zip(batch, predictions):
                postprocessed_predictions.append(vocab_processor.postprocess_tags(pred, tokens))
            predictions = postprocessed_predictions

        #### Update the number of correct tags and total tags
        #### Generate Statistics
        #### Fix tags
        for (tokens, gold), pred in zip(batch, predictions):
            index = 0
            output = []
            for token, gold_tag_label, predicted_tag in zip(tokens, gold, pred):
                if index == 0 or index == len(gold)-1:
                    index = index + 1
                    continue

                simplified_token = simplify_token(token)
                predicted_tag = int(predicted_tag)
                gold_tag = int(vocab_processor.tag2id(gold_tag_label))

                if gold_tag != vocab_processor.blank_tag_id():
                    total += 1
                    pos_gold_map[gold_tag_label] = pos_gold_map.get(gold_tag_label, 0) + 1
                    IV = vocab_processor.is_IV(simplified_token)
                    if IV:
                        IV_total += 1
                    else:
                        OOV_total += 1

                    if fix_tags:
                        predicted_tag = vocab_processor.fix_tag(params['TARGET_LANGUAGE'], predicted_tag, test_data_set)

                    predicted_tag_label = vocab_processor.id2tag(predicted_tag)

                    pos_predicted_map[predicted_tag_label] = pos_predicted_map.get(predicted_tag_label, 0) + 1

                    if gold_tag == predicted_tag:
                        match += 1
                        pos_match_map[predicted_tag_label] = pos_match_map.get(predicted_tag_label, 0) + 1
                        if IV:
                            IV_match += 1
                        else:
                            OOV_match += 1

                    if vocab_processor.convert_tag_to_petrov12(gold_tag) == vocab_processor.convert_tag_to_petrov12(predicted_tag):
                        match_petrov += 1

                    if writer:                            
                        output.append(token + '_' + vocab_processor.id2tag(predicted_tag))
                    
                index = index + 1
            if writer:
                writer.write(' '.join(output) + '\n')

    tag_info = ''
    for tag in UD_TAGS:
        tag_info += tag
        tag_info += ":" + (str(pos_gold_map[tag]) if tag in pos_gold_map else '0')
        tag_info += ":" + (str(pos_predicted_map[tag]) if tag in pos_predicted_map else '0')
        tag_info += ":" + (str(pos_match_map[tag]) if tag in pos_match_map else '0')
        tag_info += '-'
    tag_info = tag_info[:-1]
    accuracy =  0 if total == 0 else (match / total)
    accuracy_petrov = 0 if total == 0 else (match_petrov / total)

    info = tag_info + ' ' + str(match)+':'+str(match_petrov)+'/'+str(total)+'-'+str(IV_match)+'/'+str(IV_total)+'-'+str(OOV_match)+'/'+str(OOV_total)
    return loss, accuracy, accuracy_petrov, info

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    main()
