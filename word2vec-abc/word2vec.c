//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// Modified by dolaameng for learning purpose 
// Modifications include (1) simplicfication (2) more comments

// Algorithm Discussion from the project page at https://code.google.com/p/word2vec/
// * architecture: skip-gram (slower, better for infrequent words) vs CBOW (fast)
// * the training algorithm: hierarchical softmax (better for infrequent words) vs negative sampling (better for frequent words, better with low dimensional vectors)
// * sub-sampling of frequent words: can improve both accuracy and speed for large data sets (useful values are in range 1e-3 to 1e-5)
// * dimensionality of the word vectors: usually more is better, but not always
// * context (window) size: for skip-gram usually around 10, for CBOW around 5

// The generated features from the words are stored in the syn0 structure, 
// the dimensionality of the feature space is layer1_size
// So the dth feature for the cth word in vocab is syn0[c * layer1_size + d]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

// Maximum 30 * 0.7 = 21M words in the voc 
const int vocab_hash_size = 30000000; 

// Precision of float numbers
typedef float real;

struct vocab_word {
	long long cn; // count
	int * point; // ??
	char *word, *code, codelen; // ?? 
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

// vocab table
struct vocab_word *vocab;
// flags: cbow = cbow architecture
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;

int * vocab_hash;

long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

// unigram table - hashing the unigram in vocab table
int hs = 1, negative = 0;
const int table_size = 1e8;
int * table;

// initialize unigram table
void InitUnigramTable() {
	int a, i; // a for general iteration
	long long train_words_pow = 0; // total power of words_cnt in vocab
	real d1, power = 0.75;
	// allocat space for unigram table
	table = (int *)malloc(table_size * sizeof(int));
	// find the total power of word count
	// to be used as the normalization factor
	// ITERATING vocab table, NOTE vocab_size will be decided later
	for (a = 0; a < vocab_size; a++) {
		train_words_pow += pow(vocab[a].cn, power);
	}
	// d1 - the power of count of the current ref element in vocab
	i = 0;
	d1 = pow(vocab[i].cn, power) / (real)train_words_pow; 
	// ITERATING unigram table, the table_size is prefixed
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		// move to the next bin if index in vocab talbe exceeds the neighbord
		// specified by d1
		if (a / (real)table_size > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
		}
		// put everthing else in the end of the unigram table
		if (i >= vocab_size) {
			i = vocab_size - 1;
		}
	}
}

void ReadVocab() {
	//TODO
}

void LearnVocabFromTrainFile() {
	//TODO
}

void SaveVocab() {
	//TODO
}

void InitNet() {
	//TODO
}

void *TrainModelThread(void *id) {
	//TODO
}

void TrainModel(){
	long a, b, c, d;
	FILE * fo;
	// threads objects
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", train_file);

	starting_alpha = alpha;
	// build vocab either from vocab file or train file
	if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
	// save it if required
	if (save_vocab_file[0] != 0) SaveVocab();
	if (output_file[0] == 0) return;

	InitNet();

	if (negative > 0) InitUnigramTable(); // negative sampling

	// create threads to do training and block-wait
	start = clock();
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

	// write output file
	fo = fopen(output_file, "wb");
	// save word vectors
	if (classes == 0) {
		fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
		for (a = 0; a < vocab_size; a++) {
			fprintf(fo, "%s ", vocab[a].word);
			if (binary) {
				for (b = 0; b < layer1_size; b++) 
					fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
			} else {
				for (b = 0; b < layer1_size; b++)
					fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
			}
			fprintf(fo, "\n");
		}
	} else { // save the word classes
		// run kmeans on word vectors to get word classes
		int clcn = classes, iter = 0, closeid;
		// sizes of each cluster
		int *centcn = (int *)malloc(classes * sizeof(int));
		// classes of each word in vocab
		int *cl = (int *)calloc(vocab_size, sizeof(int));
		real closev, x;
		// center vector
		real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
		// initialize class labels of words in a wheel way
		for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
		// iterative training
		for (a = 0; a < iter; a++) {
			// reset centers to all zeros
			for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
			// reset cluster size to 1 element
			for (b = 0; b < clcn; b++) centcn[b] = 1;
			// for each word (for each feature of it)
			// center_vec += word_vec
			// center_size += 1
			for (c = 0; c < vocab_size; c++) {
				for (d = 0; d < layer1_size; d++) {
					// cl[c] is the cluster index of word at c
					cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
					centcn[cl[c]]++;
				}
			}
			// for each cluster (for each feature of cluster center)
			// cent_vec /= cluster_size
			// cent_vec `~ normalized by l2 norm
			for (b = 0; b < clcn; b++) {
				closev = 0;
				for (c = 0; c < layer1_size; c++) {
					// taking average
					cent[layer1_size * b + c] /= centcn[b];
					closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
				}
				// closev = l2 norm of the center vec
				// normalize the center vec by its l2 norm
				// NORMALIZATION OF CENTER VECTORS FOR LATER DISTANCE COMPARISON
				closev = sqrt(closev);
				for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
			}
			for (c = 0; c < vocab_size; c++) {
				closev = -10;
				closeid = 0;
			}
		}
		// save the kmeans classes
	}
}

// parse the command line arguments
// helper function - find the position of the argument
int ArgPos(char * str, int argc, char ** argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Augument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

// main entry
int main(int argc, char ** argv) {
	int i;
	// helper message
	if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous back of words model; default is 0 (skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
    return 0;
  }
  // initialize the file pathes to empty strings
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  // parse the arguments 
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  // allocate memory for vocab, vocab-hash, and expTable table
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  // precomputing exponetial table 
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
  	// Precompute the exp() table
  	expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
  	// Precompute f(x) = x / (x + 1)
  	expTable[i] = expTable[i] / (expTable[i] + 1);
  }
  TrainModel();
  return 0;
}