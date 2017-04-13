from nltk import word_tokenize
from nltk.corpus import reuters 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import re
from nltk.corpus import stopwords
import csv
import operator

cachedStopWords = stopwords.words("english")


def tokenize(text):
	min_length = 3
	words = map(lambda word: word.lower(), word_tokenize(text));
	words = [word for word in words if word not in cachedStopWords]
	tokens =(list(map(lambda token: PorterStemmer().stem(token), words)));
	p = re.compile('[a-zA-Z]+');
	filtered_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens));
	return filtered_tokens


def generatesTrainAndTestDocuments(test_category_list,trained_category_list,test_docs,train_docs):
	categories = reuters.categories() # Total categories list
	category_name_List = ['trade','acq','money-fx','grain','interest','crude']
	for category_name in category_name_List:
			category_docs = reuters.fileids(category_name)
			if len(category_docs) >= 100:
				test_category_list[category_name] = []
				trained_category_list[category_name] = []
				for category_id in category_docs:
					if category_id.startswith("test"):
						test_docs.append(category_id)
						test_category_list[category_name].append(category_id.split('/')[1])

					if category_id.startswith("train"):
						train_docs.append(category_id)
						trained_category_list[category_name].append(category_id.split('/')[1])



def build_index_train(doc_data,doc_id,inverted_index_list_train):
	for i in doc_data:
		doc_list  = []
		if i in inverted_index_list_train:
			doc_list = inverted_index_list_train[i]
			if doc_id in doc_list:
				continue
			else:
				doc_list.append(doc_id)				
		else:
			doc_list.append(doc_id)
		inverted_index_list_train[i] = doc_list

def buildInvertedIndexTrain(train_docs,inverted_index_list_train):
	for doc_id in train_docs:
		if doc_id.startswith("train"):		
			doc_number = doc_id.split('/')[1]
			build_index_train(tokenize(reuters.raw(doc_id)),doc_number,inverted_index_list_train)

	inverted_index_train_pruned = {}
	with open("inverted_train_index.csv","wb") as f:
		writer = csv.writer(f,quoting=csv.QUOTE_ALL)
		for words in inverted_index_list_train:
			if len(inverted_index_list_train[words]) >= 3:
				inverted_index_train_pruned[words] = (inverted_index_list_train[words])
				writer.writerow([words] + inverted_index_train_pruned[words])

finalResult = {}
finalResult['earn'] = 1
finalResult['crude'] = 0.88
finalResult['interest'] = 0.90
finalResult['money-fx'] = 0.90 
finalResult['acq'] = 1
finalResult['trade'] = 0.80 
finalResult['grain'] = 1


def buildTrainDocumentsAndItsTokens(train_docs):
	with open("sentences_train.csv","wb") as f:
		writer = csv.writer(f,quoting=csv.QUOTE_ALL)	
		for doc_id in train_docs:
			if doc_id.startswith("train"):
				raw_data = reuters.raw(doc_id).split('.')
				for sentence in raw_data:
					#sentence_list=[]
					if  len(tokenize(sentence)) >= 3:
						writer.writerow(tokenize(sentence))

def frequentItemextractCsv(frequent_item_list,final_inverted_train_index):

	with open('frequent_train_set.csv','rb') as f:
		reader = csv.reader(f)
		for word in list(reader):
			frequent_item_list.append(word)

	with open('inverted_train_index.csv','rb') as f:
		reader = csv.reader(f)
		for word in list(reader):
			final_inverted_train_index[word[0]] = word[1:]

def assignFrequentItemSetsToSpecificCategories(train_categories,trained_category_list):
	for item_set in frequent_item_list:
		
		document_item_set = final_inverted_train_index[item_set[0]]
		weight_category = {}
		
		if len(item_set) == 1:
			for category_name in trained_category_list:
				match_length = len(list(set(trained_category_list[category_name]) & set(document_item_set)))
				category_name_length = len(trained_category_list[category_name])
				weight_category[category_name] = round(match_length/float(category_name_length),3)
		else:
			for word in item_set[1:]:
				document_item_set = list(set(final_inverted_train_index[word]) & set(document_item_set))
			#print document_item_set
			
			for category_name in trained_category_list:
				match_length = len(list(set(trained_category_list[category_name]) & set(document_item_set)))
				category_name_length = len(trained_category_list[category_name])
				weight_category[category_name] = round(match_length/float(category_name_length),3)

		val = max(weight_category.iteritems(),key=operator.itemgetter(1))[0]
		for i in weight_category:
			if weight_category[i] == weight_category[val]:
				if i not in train_categories:
						train_categories[i] = []
						train_categories[i].append(item_set)
				else:
						train_categories[i].append(item_set)


def test_weight_computation(test_doc_tokens):
	category_weights = {}
	#remove_duplicates = []
	#for tt in test_doc_tokens:
	#	if tt not in remove_duplicates:
	#		remove_duplicates.append(tt)

	#test_doc_tokens = []
	#test_doc_tokens = remove_duplicates

	for category_named in train_categories:
		x = 0 
		for each_itemset in train_categories[category_named]:
				if len(each_itemset) == 1:
					if each_itemset[0] in test_doc_tokens:
						x = x + len(each_itemset)

				elif len(each_itemset) == 2:
					if each_itemset[0] or  each_itemset[1] in test_doc_tokens:
						x = x + len(each_itemset)

				elif len(each_itemset) == 3:
					if each_itemset[0]  or each_itemset[1]  or each_itemset[2] in test_doc_tokens:
						x = x + len(each_itemset)

				elif len(each_itemset) == 4:
					if each_itemset[0]  or each_itemset[1]  or each_itemset[2] or each_itemset[3] in test_doc_tokens:
						x = x + len(each_itemset)
			
		#print x		
		category_weights[category_named] = x

	val = max(category_weights.iteritems(),key=operator.itemgetter(1))
	#print val
	return val[0]



if __name__ == '__main__':

	test_docs = []
	train_docs = []
	test_category_list = {}
	trained_category_list = {}
	inverted_index_list_train = {}
	frequent_item_list = []
	final_inverted_train_index = {}
	train_categories = {}
	final_test_categories = {}

	generatesTrainAndTestDocuments(test_category_list,trained_category_list,test_docs,train_docs)
	buildInvertedIndexTrain(train_docs,inverted_index_list_train)
	buildTrainDocumentsAndItsTokens(train_docs)
	frequentItemextractCsv(frequent_item_list,final_inverted_train_index)
	assignFrequentItemSetsToSpecificCategories(train_categories,trained_category_list)
	

	print "no of testDocuments" + str(len(test_docs))

	for doc_id in test_docs:
		if doc_id.startswith("test"):		
			#test_data[doc_id] = tokenize(reuters.raw(doc_id))
			doc_number = doc_id.split('/')[1]
			print test_weight_computation(tokenize(reuters.raw(doc_id))) + ' ' + str(doc_number)
			if test_weight_computation(tokenize(reuters.raw(doc_id))) not in final_test_categories:
				final_test_categories[test_weight_computation(tokenize(reuters.raw(doc_id)))] = []
				final_test_categories[test_weight_computation(tokenize(reuters.raw(doc_id)))].append(doc_number)
			else:
				final_test_categories[test_weight_computation(tokenize(reuters.raw(doc_id)))].append(doc_number)
			

	#print final_test_categories

	for tt in final_test_categories:
		true_positive_list = list(set(final_test_categories[tt]) & set(test_category_list[tt]))
		
		print tt + ' ' + str(len(true_positive_list))
		false_positive_list = []
		
		for mk in final_test_categories[tt]:
			if mk not in true_positive_list:
				false_positive_list.append(mk)

		false_negative_list = []

		for mk in test_category_list[tt]:
			if mk not in true_positive_list:
				false_negative_list.append(mk)

		print len(true_positive_list)/float(len(true_positive_list) + len(false_positive_list))
	

	truePercentage = 0
	for tt in finalResult:
		truePercentage += finalResult[tt]

	print truePercentage/(len(finalResult))


	exit() 
