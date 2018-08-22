import pandas as pd 
import numpy as np 
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re

def load_data(path, max_len = None):
  #load text and labels from files, and clean texts
  data = pd.read_csv(path)
  question1 = data["question1"].fillna("_na_").values
  question2 = data["question2"].fillna("_na_").values
  is_duplicate = data["is_duplicate"].values
  qid1 = data["qid1"].values
  qid2 = data["qid2"].values

  processed_question1 = []
  processed_question2 = []
  label = []
  for i in range(len(question1)):
    cleaned_q1 = clean_str(str(question1[i]))
    cleaned_q2 = clean_str(str(question2[i]))
    if (len(cleaned_q1)==1 or len(cleaned_q1)==0):
      continue
    if (max_len != None):
      if (len(cleaned_q1)>60 or len(cleaned_q2)>60):
        continue

    processed_question1.append(cleaned_q1)
    processed_question2.append(cleaned_q2)
    label.append([1,0] if is_duplicate[i] == 0 else [0,1]) 

  all_questions = processed_question2+processed_question1
  word2id, word_embeddings = create_word_id_map(all_questions)
  return processed_question1, processed_question2, label, word2id, word_embeddings

def clean_str(string):
  #clean texts
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)#移除非字母非数字的字符
  string = re.sub(r"\'s", " 's", string)
  string = re.sub(r"\'ve", " 've", string)
  string = re.sub(r"n\'t", " n't", string)
  string = re.sub(r"\'re", " 're", string)
  string = re.sub(r"\'d", " 'd", string)
  string = re.sub(r"\'ll", " 'll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " ( ", string)
  string = re.sub(r"\)", " ) ", string)
  string = re.sub(r"\?", " ? ", string)
  string = re.sub(r"\s{2,}", " ", string)

  cleaned = string.strip()
  cleaned = cleaned.lower()
  lemmer = WordNetLemmatizer()
  lemmed = [lemmer.lemmatize(w) for w in cleaned.split()]

  return lemmed

def create_word_id_map(data):
  #build vocabulary from all training texts
  #create pretrained word embeddings and word2id dictionary to convert words to int
  counter = Counter()
  for question in data:
    counter.update(question)
  mim_word_counts = 100
  vocab = [word for word, freq in counter.items() if freq>mim_word_counts]
  vocab = ["PAD","UNK"]+vocab
  word2id = {word:id for id, word in enumerate(vocab)}

  glove_dict = load_Glove("../glove.6B/glove.6B.200d.txt")
  word_embeddings = create_wordembeddings(glove_dict, word2id)
  print ("in total %d words in our texts" %len(counter))
  print ("finally %d words in our built vocabulary" %len(vocab))
  

  return word2id , word_embeddings


def load_Glove(glove_path):
  glove_dict = {}
  f = open(glove_path, 'r', encoding="utf8")
  for line in f:
    splitline = line.split()
    word = splitline[0]
    word_vector =np.array([float(val) for val in splitline[1:]])
    glove_dict[word] = word_vector
  return glove_dict

def create_wordembeddings(glove_dict, word2id):
    
  all_embs = np.stack(glove_dict.values())
  emb_mean,emb_std = all_embs.mean(), all_embs.std()
  vocab_size = len(word2id)
  embedding_size = all_embs.shape[1]
  word_embeddings = np.random.normal(emb_mean, emb_std, [vocab_size, embedding_size])
  for word, id in word2id.items():
    word_vec = glove_dict.get(word)
    if word_vec is not None:
      word_embeddings[id] = word_vec
  return word_embeddings
  
def pad_seq_id(question1, question2, word2id, max_seq_len = None):
  #pad sentences to same lengths
  #convert words to ints in word2id dict 
  max_len1 = max([len(q) for q in question1])
  max_len2 = max([len(q) for q in question2])

  if (max_seq_len == None):
    sentence_length = max(max_len1, max_len2)
  else:
    sentence_length = max_seq_len
  sentence_num = len(question1)
  
  question1_id = np.zeros([sentence_num, sentence_length], dtype=int)
  question2_id = np.zeros([sentence_num, sentence_length], dtype=int)

  for i,s in enumerate(question1):
    question1_id[i][:len(s)] = [word2id.get(w, word2id["UNK"]) for w in s]
  for i ,s in enumerate(question2):
    question2_id[i][:len(s)] = [word2id.get(w, word2id["UNK"]) for w in s]

  return question1_id, question2_id

def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    the input data for this batch iterator is already processed to be in same length
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    print ("num_batches_per_epoch:", num_batches_per_epoch)
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]

def batch_iter_pad(data, batch_size, word2id, shuffle=True):
  """
  Generated a batch iterator for a dataset
  in this iterator, the generated batches have different lengths
  but in one batch, the lengths are the same
  """
  data_size = len(data)
  num_batches_per_epoch = int((len(data)-1)/batch_size)+1
  print ("num_batches_per_epoch：", num_batches_per_epoch)
  if shuffle:
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_data = [data[i] for i in shuffle_indices]
  else:
    shuffled_data = data
  if len(list(zip(*data))) == 3:
    question1, question2, label = zip(*data)
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num*batch_size
      end_index =  min((batch_num + 1) * batch_size, data_size)
      label_batch = label[start_index:end_index]
      question1_len = [len(s) for s in question1[start_index:end_index]]
      question2_len = [len(s) for s in question2[start_index:end_index]]
      question1_id, question2_id = pad_seq_id(question1[start_index:end_index], question2[start_index:end_index], word2id)
      yield list(zip(question1_id, question2_id, question1_len, question2_len, label_batch))
  else:
    question1, question2 = zip(*data)
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num*batch_size
      end_index =  min((batch_num + 1) * batch_size, data_size)
      question1_len = [len(s) for s in question1[start_index:end_index]]
      question2_len = [len(s) for s in question2[start_index:end_index]]
      question1_id, question2_id = pad_seq_id(question1[start_index:end_index], question2[start_index:end_index], word2id)
      yield list(zip(question1_id, question2_id, question1_len, question2_len))


def shuffle_data(data):
  num = len(data)
  ids = list(range(num))
  random.shuffle(ids) #shuffle will change the list in-place
  return [data[i] for i in ids]



if __name__ == "__main__":
  load_data("inputs/train.csv")






