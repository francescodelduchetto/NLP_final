''' '''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
#from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
#from keras.datasets import imdb

from gensim.models.keyedvectors import KeyedVectors
import nltk
import os
import numpy as np

# One-hot
tokenizer = Tokenizer()

# LSTM
lstm_output_size = 128

# Training
batch_size = 30
epochs = 2

# Pos Tag enum (31 two times bcs it tags lso with # and dunno what it is)
POS_TAGS = {
  "$": 0, "''": 1, "(": 2, ")": 3, ",": 4, "--": 5, ".": 6, ":": 7, "CC": 8, "CD": 9,
  "DT": 10, "EX": 11, "FW": 12, "IN": 13, "JJ": 14, "JJR": 15, "JJS": 16, "LS": 17,
  "MD": 18, "NN": 19, "NNP": 20, "NNPS": 21, "NNS": 22, "PDT": 23, "POS": 24, "PRP": 25,
  "PRP$": 26, "RB": 27, "RBR": 28, "RBS": 29, "RP": 30, "SYM": 31, "#": 31, "TO": 32, "UH": 33,
  "VB": 34, "VBD": 35, "VBG": 36, "VBN": 37, "VBP": 38, "VBZ": 39, "WDT": 40, "WP": 41,
  "WP$": 42, "WRB": 43, "``": 44
}

# Relations enum
RELATIONS = {
  "COLOR": 0, "MATERIAL": 1, "PLACE": 2, "GENERALIZATION": 3, "SIZE": 4, "HOW_TO_USE": 5, 
  "SPECIALIZATION": 6, "PART": 7, "ACTIVITY": 8, "SHAPE": 9, "SIMILARITY": 10, "PURPOSE": 11, 
  "SOUND": 12, "TASTE": 13, "SMELL": 14, "TIME": 15
}

# Max len input sequence
max_len = 30

# Where to find data
resource_dir = "data/"

# W2V resources
word_vectors = None
random_vector = None

def getPosTag(tok_sentence):
  pos_tagged = nltk.pos_tag(tok_sentence)
  _, tags = zip(*pos_tagged)
  tag_ids = []
  max_value = max(POS_TAGS.values())
  mean_value = max_value / 2.0
  for tag in tags:
    tag_ids.append((POS_TAGS[tag] - mean_value) / float(mean_value))

  return tags, tag_ids

def w2vAndPadData(data):
  w2v_padded_data = np.zeros((len(data), max_len, 301), dtype=np.float64)

  for s_idx, sentence in enumerate(data):
    w2v_sent = np.zeros((max_len, 301), dtype=np.float64) # fill until max len
    _, tag_ids = getPosTag(sentence)
    for w_idx in range(max_len):
      try:
        w2v_sent[w_idx] = np.append(word_vectors[sentence[w_idx]], tag_ids[w_idx])
      except KeyError: # we put the UNK vector if not in w2v or sentence finished
        w2v_sent[w_idx] = np.append(random_vector, tag_ids[w_idx])
      except IndexError:
        w2v_sent[w_idx] = np.append(random_vector, POS_TAGS["."])
    w2v_padded_data[s_idx] = w2v_sent

  return w2v_padded_data

def relToVector(relations):
  max_rel_value = len(RELATIONS.values())
  vec_rel = np.zeros((len(relations), max_rel_value), dtype=np.float64)

  for idx, relation in enumerate(relations):
    vec_rel[idx][RELATIONS[relation]] = 1

  return vec_rel
  
def processData(sent_file, rel_file, num_samples):
  sents = []
  rels = []
  for _ in range(num_samples):
    question = sent_file.readline()
    if question == '':
      break
    sents.append(nltk.word_tokenize(question.decode('utf8')))
    rels.append(rel_file.readline().replace('\n',''))

  x = w2vAndPadData(sents)
  y = relToVector(rels)
  
  return x, y

if __name__ == "__main__":

  print('Loading data...')
  ## Open all the needed files
  x_train_file = open("data/train/questions.txt", "r")
  rel_train_file = open("data/train/rels.txt", "r")
  x_test_file = open("data/test/questions.txt", "r")
  rel_test_file = open("data/test/rels.txt", "r")
  x_dev_file = open("data/dev/questions.txt", "r")
  rel_dev_file = open("data/dev/rels.txt", "r")

  #print("Example train item:", sent_train[0])
  #print("Example train label:", c1_train[0], c2_train[0])
  #print("Example test item:", sent_test[34])
  #print("Example test label:", c1_test[34], c2_test[34])
  
 # assert(len(sent_train)==len(c1_train)==len(c2_train))
 # assert(len(sent_test)==len(c1_test)==len(c2_test))

  ## Loading word2vec
  try:
    word_vectors = KeyedVectors.load(resource_dir + "wv.w2v", mmap='r')
    print("Loaded Word2Vec from the saved model")
  except IOError:
    print('Loading W2V..')
    wv = KeyedVectors.load_word2vec_format(resource_dir + 'GoogleNews-vectors-negative300.bin', binary=True)
    wv.save(resource_dir + "wv.w2v")
    word_vectors = KeyedVectors.load(resource_dir + "wv.w2v", mmap='r')
  ## Random vector for word out of w2v
  try:
    random_vector = np.load(resource_dir + "UNK_vec.npy")
    print("Loaded random UNK vector from the saved model")
  except IOError:
    print("No random UNK vector found, retrain your model with this random vector!")
    random_vector = np.random.rand(300)
    np.save(resource_dir + "UNK_vec", random_vector)

  ## Build Model
  print('Build model...')

  model = Sequential()
  model.add(LSTM(lstm_output_size,
                   input_shape=(max_len, 301)))
  model.add(Dense(len(RELATIONS.values())))
  model.add(Activation('sigmoid'))

  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  ## Train loop
  while True:
    print("## new step") 
    ## Process train/dev data 
    x_train, y_train = processData(x_train_file, rel_train_file, 10000)
    x_dev, y_dev = processData(x_dev_file, rel_dev_file, 2000)

    #print(x_train[100], y_train[100])

    if x_train.shape[0] == 0 or x_dev.shape[0] == 0:
      break

    print(x_train.shape, 'train sequences')
    print(x_dev.shape, 'dev sequences')

    ## Train 
    print('Train...')
    model.fit(x_train, 
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_dev, y_dev))

  ## Evaluate on test
  x_test, y_test = processData(x_test_file, rel_test_file, 100000)

  score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
  print('Test score:', score)
  print('Test accuracy:', acc)


  ## Save model to json
  os.system("mkdir model_rel_test")
  model_json = model.to_json()
  with open("model_rel_test/keras_model.json", "w") as model_file:
    model_file.write(model_json)
  ## Save weights to hdf5
  model.save_weights("model_rel_test/model_weights.hd")
  print("Model saved")

