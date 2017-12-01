#ML-Tensorflow-ProcessData.py
#For process DATA ONLY, neuralNetwork.py is for trainning and testing
import nltk, random, pickle, time
from nltk.tokenize import word_tokenize
import numpy as np 
from collections import Counter
from nltk.stem import WordNetLemmatizer
import pandas as pd
'''
polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def init_process(fin, fout):
	outfile = open(fout, 'a')
	
	with open(fin, buffering = 200000, encoding = 'latin-1') as f:
		try:
			for line in f:
				line = line.replace('"', '')
				initial_polarity = line.split(',')[0]
				#classification
				if initial_polarity == '0':
					initial_polarity = [1,0]
				elif initial_polarity == '4':
					initial_polarity = [0,1]

				tweet = line.split(',')[-1]
				outline = str(initial_polarity) + ':::' + tweet
				
				outfile.write(outline)
				
		except Exception as e:
			print(str(e))
	outfile.close()
#keep polarity and tweet
init_process('training.1600000.processed.noemoticon.csv','train_set.csv')
init_process('testdata.manual.2009.06.14.csv','test_set.csv')


#生成样本字典，将文本转化为纯单词的list，作为样本字典
#把tweet里面所有单词作为lexicon
def create_lexicon(fin):
	lexicon = []

	with open(fin, 'r', buffering = 100000, encoding = 'latin-1') as f:
		try:
			counter = 1
			content = ''
			for line in f:
				counter += 1
				if (counter/2500.00).is_integer():
					tweet = line.split(':::')[1]
					content += ' '+tweet
					words = word_tokenize(content)
					words = [lemmatizer.lemmatize(i) for i in words]
					lexicon = list(set(lexicon + words))
			print(counter, len(lexicon))
		except Exception as e:
			print(str(e))

	with open('lexicon.pickle', 'wb') as f:
		pickle.dump(lexicon, f)

create_lexicon('train_set.csv')

#找出test里面的单词在lexicon中的次数
def convert_to_vec(fin, fout, lexicon_pickle):
	with open(lexicon_pickle, 'rb') as f:
		lexicon = pickle.load(f)

	# Open for writing.  The file is created if it does not exist
	outfile = open(fout, 'a')
	with open(fin, buffering = 20000, encoding = 'latin-1') as f:
		counter = 0
		for line in f:
			counter += 1

			label = line.split(':::')[0]
			tweet = line.split(':::')[1]
			current_words = word_tokenize(tweet.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))

			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					# OR DO +=1, test both
					features[index_value] += 1

			features = list(features)
			outline = str(features) + '::' + str(label) + '\n'
			outfile.write(outline)

		print(counter, 'counter in convert_to_vec')

convert_to_vec('test_set.csv','processed-test-set.csv','lexicon.pickle')

def shuffle_data(fin):
	df = pd.read_csv(fin, error_bad_lines=False)
	df = df.iloc[np.random.permutation(len(df))]
	print(df.head())
	df.to_csv('train_set_shuffled.csv', index=False)
	
shuffle_data('train_set.csv')

def create_test_data_pickle(fin):
	feature_sets = []
	labels = []
	counter = 0
	with open(fin, buffering = 20000) as f:
		for line in f:
			try:
				features = list(eval(line.split('::')[0]))
				label = list(eval(line.split('::')[1]))
				feature_sets.append(features)
				labels.append(label)
				counter += 1
			except Exception as e:
				pass
	print(counter,'counter in create_test_data_pickle')
	feature_sets = np.array(feature_sets)
	labels = np.array(labels)

create_test_data_pickle('processed-test-set.csv')
	

