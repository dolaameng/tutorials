# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ## pyword2vec single thread experiment

# <codecell>

import numpy as np
from collections import Counter
import mmap
import re
import networkx as nx
import math

# <codecell>

## HELPER FUNCTION
def file_to_wordstream(file_path):
    with open(file_path) as fin:
        mf = mmap.mmap(fin.fileno(), 0, access = mmap.ACCESS_READ)
        for match in re.finditer(r'(.*?)\s', mf):
            word = match.group(1)
            if word:
                yield word
                
def inspect_vocab_tree(vocab):
    g = nx.DiGraph()
    vocab_size = len(vocab)
    edges = set()
    for vw in vocab:
        tree_path = [i + vocab_size for i in vw['path']]
        tree_path = [str(i) if i >= vocab_size 
                         else "%d_%s(%d)" % (i, vocab[i]['word'], vocab[i]['count']) 
                     for i in tree_path]
        edges.update(zip(tree_path[:-1], tree_path[1:]))
    g.add_edges_from(edges)
    figure(figsize=(16, 16))
    pos = nx.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=True, arrows = True, node_size=3000, font_size = 30)
    return g

# <codecell>

## HashedVocab

class HashedVocab(object):
    def __init__(self):
        self.HASH_SIZE = 30000000
        self.MIN_COUNT = 5
        self.vocab = []
        self.vocab_hash = np.empty(self.HASH_SIZE, dtype = np.int)
        self.vocab_hash.fill(-1)
    def fit(self, word_stream):
        counter = Counter(word_stream)
        self.vocab = map(lambda x: dict(zip(['word', 'count'], x)), 
                         filter(lambda x: x[1] > self.MIN_COUNT, 
                                    counter.most_common(len(counter))))
        if len(self.vocab) > self.HASH_SIZE * 0.7:
            raise RuntimeError('Vocab size too large, increase MIN_COUNT or increase HASH_SIZE')
        self.build_hash()
        self.build_huffman_tree()
    def search_for(self, word):
        word_hash = self.get_word_hash(word)
        while True:
            word_index = self.vocab_hash[word_hash]
            if word_index == -1:
                return - 1
            elif word == self.vocab[word_index]['word']:
                return word_index
            else:
                word_hash = (word_hash + 1) % self.HASH_SIZE
    def __getitem__(self, word_index):
        return self.vocab[word_index]
    
    def build_hash(self):
        self.vocab_hash = np.empty(self.HASH_SIZE, dtype = np.int)
        self.vocab_hash.fill(-1)
        for word_index, vocab_word in enumerate(self.vocab):
            word = vocab_word['word']
            word_hash = self.get_word_hash(word)
            self.add_to_hash(word_hash, word_index)
    def get_word_hash(self, word):
        ## TOO SLOW
        #word_hash = sum([ord(c)*(257**i) 
        #             for i, c in zip(range(len(word))[::-1], word)])
        word_hash = 0
        for c in word:
            word_hash = word_hash * 257 + ord(c)
        word_hash %= self.HASH_SIZE
        return word_hash
    def add_to_hash(self, word_hash, word_index):
        while self.vocab_hash[word_hash] != -1:
            word_hash = (word_hash + 1) % self.HASH_SIZE
        self.vocab_hash[word_hash] = word_index
    def build_huffman_tree(self):
        """
        build binary Huffman Tree by word counts,
        the structure will be embedded in the vocab_word
        dict(word, count, path, code) in the vocab
        
        vocab_word['code'] will be the binary representation of word
        based on frequency
        vocab_word['path'] will be the path from root to leaf 
        """
        ## for arbitary full binary, n-1 internal inodes at max 
        ## given n leaves. But in the original C code, the count
        ## binary and parent_node size are n*2+1 instead of n*2-1
        vocab_size = len(self.vocab)
        print "DEBUG:", vocab_size
        ## workhorse data structure for tree construction
        count = np.empty(vocab_size*2-1, dtype = np.int64)
        count.fill(1e15)
        count[:vocab_size] = [vw['count'] for vw in self.vocab]
        ## boolean values of each node
        binary = np.zeros(vocab_size*2-1, dtype = np.int8)
        ## parent node to store the path
        parent_node = np.empty(vocab_size*2-1, dtype = np.int64)
        ## construct the tree
        pos1, pos2 = vocab_size-1, vocab_size
        for a in xrange(vocab_size-1):
            ## min1i
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1i, pos1 = pos1, pos1-1
                else:
                    min1i, pos2 = pos2, pos2+1
            else:
                min1i, pos2 = pos2, pos2+1
            ## min2i
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2i, pos1 = pos1, pos1-1
                else:
                    min2i, pos2 = pos2, pos2+1
            else:
                min2i, pos2 = pos2, pos2+1
            count[vocab_size + a] = count[min1i] + count[min2i]
            parent_node[min1i] = vocab_size + a
            parent_node[min2i] = vocab_size + a
            binary[min2i] = 1
        for a in xrange(vocab_size):
            b, i = a, 0
            code, path = [], []
            while True:
                code.append(binary[b])
                path.append(b)
                i += 1
                b = parent_node[b]
                if b == vocab_size * 2 - 2: break
            self.vocab[a]['path'] = [vocab_size - 2] + [p -vocab_size for p in path[::-1]]
            self.vocab[a]['code'] = code[::-1]

# <codecell>

## Learning Models
class Word2Vec(object):
    """
    Net Architecture: cbow / skip_gram
    Learning: hs / negative_sampling
    """
    def __init__(self, hashed_vocab, layer1_size,
                 net_type, learn_type):
        """
        hashed_vocab: vocab to build on
        layer1_size: the dimensionality of feature space
        net_type: {'cbow', 'skip_gram'}
        learn_type : {'hs', 'negative'}
        """
        self.vocab = hashed_vocab
        self.layer1_size = layer1_size
        self.net_type = net_type
        self.learn_type = learn_type
        
        self.starting_alpha = 0.025
        self.sentence_len = 1000
        self.sampling = 1e-4
        
        self.REAL_TYPE = np.float64
        self.syn0 = np.array([], dtype = self.REAL_TYPE)
        self.syn1 = np.array([], dtype = self.REAL_TYPE)
        self.syn1neg = np.array([], dtype = self.REAL_TYPE)
    def init_net(self):
        """
        syn0 - len(vocab) * layer1_size
        syn1 - len(vocab) * layer1_size
        syn1neg - len(vocab) * layer1_size
        """
        vocab_size = len(self.vocab)
        self.syn0 = np.random.uniform(low = -.5 / self.layer1_size, 
                                      high = .5 / self.layer1_size, 
                                      size = (vocab_size, self.layer1_size)).astype(self.REAL_TYPE)
        if self.learn_type == 'hs': 
            self.syn1 = np.empty((vocab_size, self.layer1_size), dtype = self.REAL_TYPE)
        elif self.learn_type == 'negative':
            self.syn1neg = np.empty((vocab_size, self.layer1_size), dtype = self.REAL_TYPE)
    def fit(self, words):
        """
        word_stream: stream of words
        ntotal: total number of words in word_stream
        """
        neu1 = np.empty(self.layer1_size, dtype = self.REAL_TYPE)
        neu1e = np.empty(self.layer1_size, dtype = self.REAL_TYPE)
        ntotal = len(words)
        
        alpha = self.starting_alpha
        next_random = 0
        
        nprocessed = 0
        while nprocessed < ntotal:
            ## adjust learning rate alpha
            alpha = max(self.starting_alpha * (1 - nprocessed / (ntotal + 1.)), 
                        self.starting_alpha * 0.0001)
            ## refill the sentence
            sentence_index = []
            while nprocessed < ntotal and len(sentence_index) < self.sentence_len:
                ## sampling down the infrequent words
                if nprocessed % 100000 == 0: 
                    print 'progress:', nprocessed * 100. / ntotal, '%'
                word = words[nprocessed]
                word_index = self.vocab.search_for(word)
                word_count = self.vocab[word_index]['count']
                nprocessed += 1
                if word_index == -1: continue
                if self.sampling > 0:
                    ran = ( (math.sqrt(word_count / (self.sampling * ntotal)) + 1) 
                                                       * (self.sampling * ntotal) / word_count);
                    next_random = next_random * 25214903917 + 11
                    if ran < (next_random & 0xFFFF) / 65536.: continue
                sentence_index.append(word_index)

# <markdowncell>

# ##TESTING

# <codecell>

## TEST HashedVocab
hashed_vocab = HashedVocab()
#hashed_vocab.fit(file_to_wordstream('data/text_simple'))
hashed_vocab.fit(file_to_wordstream('data/simple.txt'))

# <codecell>

for i, vw in enumerate(hashed_vocab.vocab):
    word = vw['word']
    assert i == hashed_vocab.search_for(word)
    
print hashed_vocab.search_for('alien')
print len(hashed_vocab.vocab)
#inspect_vocab_tree(hashed_vocab.vocab)

# <codecell>

train_words = list(file_to_wordstream('data/simple.txt'))

# <codecell>

word2vec = Word2Vec(hashed_vocab, 100, 'cbow', 'hs')
word2vec.fit(train_words)

# <codecell>


