from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from nltk.tokenize import sent_tokenize
import nltk 
import numpy as np
from nltk.tokenize import sent_tokenize
import pandas as pd
from nltk.corpus import stopwords 
stop_words = stopwords.words('english')
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx 
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
import numpy as np 
from nltk.corpus import stopwords 
stop_words = stopwords.words('english')
from .models import *
import re
import plotly.graph_objects as go
import json
import pandas as pd
import nltk
from os import listdir
from sklearn.metrics.pairwise import cosine_similarity
from os.path import isfile, join
import seaborn as sns
from tqdm import tqdm
import glob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import string
from nltk.stem import WordNetLemmatizer
from pattern.en import tag
from nltk.corpus import wordnet as wn
import warnings
import scipy.sparse as sp 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from nltk.corpus import reuters
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk.data
import math
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
import scipy as sc
stop_words = stopwords.words('english')
ideal_sent_length = 20.0
stemmer = SnowballStemmer("english")
import re
from urllib.request import urlopen
#import gensim 
import numpy as np 
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import re
import spacy  # For preprocessing

import logging  # Setting up the loggings to monitor gensim

import spacy

from tqdm import tqdm
import seaborn as sb
from matplotlib import pyplot as plt

wnl = WordNetLemmatizer()


SUMMARY_LENGTH = 12  # number of sentences in final summary
stop_words = stopwords.words('english')

from rank_bm25 import BM25Okapi
import pandas as pd
data = pd.read_csv("test.csv")
from rank_bm25 import BM25Okapi
english_stopwords = list(set(stopwords.words('english')))

def strip_characters(text):
    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<', '', text)
    t = re.sub('/', ' ', t)
    t = t.replace("'",'')
    return t

def clean(text):
    t = text.lower()
    t = strip_characters(t)
    return t

def tokenize(text):
    words = nltk.word_tokenize(text)
    return list(set([word for word in words 
                     if len(word) > 1
                     and not word in english_stopwords
                     and not (word.isnumeric() and len(word) is not 4)
                     and (not word.isnumeric() or word.isalpha())] )
               )

def preprocess(text):
    t = clean(text)
    tokens = tokenize(t)
    return tokens

class SearchResults:
    
    def __init__(self, 
                 data: pd.DataFrame,
                 columns = None):
        self.results = data
        if columns:
            self.results = self.results[columns]
        #print( "self.results")
        #print( self.results)
            
    def __getitem__(self, item):
        return Paper(self.results.loc[item])
    
    def __len__(self):
        return len(self.results)
        
    def _repr_html_(self):
        return self.results._repr_html_()

SEARCH_DISPLAY_COLUMNS = ['paper_id', 'title', 'abstract', 'body_text']

class WordTokenIndex:
    
    def __init__(self, 
                 corpus: pd.DataFrame, 
                 columns=SEARCH_DISPLAY_COLUMNS):
        self.corpus = corpus
        #print("self.corpus")
        #print(self.corpus)
        raw_search_str = self.corpus.abstract.fillna('') + ' ' + self.corpus.title.fillna('') + ' ' + self.corpus.body_text.fillna('')
        #print("raw_search_str")
        #print(raw_search_str)
        self.index = raw_search_str.apply(preprocess).to_frame()
        #print( "self.index")
        #print( self.index)
        #print( self.index[0])
        self.index.columns = ['terms']
        #print("self.index.columns")
        #print(self.index.columns)
        self.index.index = self.corpus.index
        #print("self.index.index")
        #print(self.index.index)
        self.columns = columns
        #print("self.columns")
        #print(self.columns)
        return self.columns
    
    def search(self, search_string):
        search_terms = preprocess(search_string)
        #print("search_terms" )
        #print(search_terms )
        result_index = self.index.terms.apply(lambda terms: any(i in terms for i in search_terms))
        #print("result_index")
        #print(result_index )
        results = self.corpus[result_index].copy().reset_index().rename(columns={'index':'paper'})
        #print("results")
        #print(results)
        #print("SearchResults(results, self.columns + ['paper'])")
        #print(SearchResults(results, self.columns + ['paper']))
        return SearchResults(results, self.columns + ['paper'])
class RankBM25Index(WordTokenIndex):
        
    def __init__(self, corpus: pd.DataFrame, columns=SEARCH_DISPLAY_COLUMNS):
        super().__init__(corpus, columns)
        #print("self.index.terms.tolist()")
        #print(self.index.terms.tolist())
        self.bm25 = BM25Okapi(self.index.terms.tolist())
        #print("self.bm25")
        #print(self.bm25)
        
    def search(self, search_string):
        search_terms = preprocess(search_string)
        doc_scores = self.bm25.get_scores(search_terms)
        #print('doc_scores')
        #print(doc_scores)
        ind = np.argsort(doc_scores)[::-1]
        #print('ind')
        #print(ind)
        results = self.corpus.iloc[ind][self.columns]
        #print('results')
        #print(results)
        results['Score'] = doc_scores[ind]
        #print("results['Score']")
        #print(results['Score'])
        results['orig_ind'] = ind
        results['word'] = search_string
        #print("results['orig_ind']")
        #print(results['orig_ind'])
        results = results[results.Score > 0]
        #print("results")
        #print(results)
        return SearchResults(results.reset_index(), self.columns + ['Score', 'orig_ind','word'])




class Summarizer():

    def penn_to_wn(self,tag):
        
        if tag.startswith('N'):
            return 'n'
 
        if tag.startswith('V'):
        
            return 'v'
 
        if tag.startswith('J'):
            return 'a'
 
        if tag.startswith('R'):
            return 'r'
        return None 
 

    def tagged_to_synset(self,word, tag):
        wn_tag = self.penn_to_wn(tag)
    
        if wn_tag is None:
            return None
 
        try:
        
            return wn.synsets(word, wn_tag)[0]
        except:
            return None
 
    def sentence_similarity(self,sentence1, sentence2):
       
        # Tokenize and tag
        sentence1 = pos_tag(word_tokenize(sentence1))
        sentence2 = pos_tag(word_tokenize(sentence2))
      
 
        # Get the synsets for the tagged words
        synsets1 = [self.tagged_to_synset(*tagged_word) for tagged_word in sentence1]
        
        synsets2 = [self.tagged_to_synset(*tagged_word) for tagged_word in sentence2]
       
 
    # Filter out the Nones
        synsets1 = [ss for ss in synsets1 if ss]
      
        synsets2 = [ss for ss in synsets2 if ss]
        
 
        score, count = 0, 0
        for synset in synsets1:
            max_sim = 0.0
            maxx=0
            for ss in synsets2:
                
                sim=wn.wup_similarity(synset, ss)
                if sim is not None and sim > max_sim:
                
                       max_sim = sim
                   
           
        
            if max_sim is not None and max_sim!=0:
                score += max_sim
                count += 1
 
    # Average the values
        if count!=0:
            score /= count
            return score   
    def __init__(self, article):
           
            self._articles = []
            #i=1
           
            for row in article:
                #if i<=20:
                    title=row[3]
                    #print(title)
                    body=row[2].replace('\n', ' ')
                    #print(body)
                    paper_id=row[0]
                    #doi=row[0]
                    if title!=''and body!='':
                        self._articles.append((paper_id,title,body))
                    #i=i+1
            
            
    def valid_input(self, headline, article_text):
        return headline != '' and article_text != ''    
    def normalize_corpus(self,corpus, lemmatize=True):
        
        normalized_corpus = []    
        for text in corpus:
            if lemmatize:
                text = self.lemmatize_text(text)
            else:
                text = text.lower()
            text = self.remove_special_characters(text)
            text = self.remove_stopwords(text)
            normalized_corpus.append(text)
        return normalized_corpus
    def pos_tag_text(self,text):
        wnl = WordNetLemmatizer()
        tagged_text = tag(text)
        tagged_lower_text = [(word.lower(), self.penn_to_wn(pos_tag))
                             for word, pos_tag in
                             tagged_text]
        return tagged_lower_text
        
         
    def score(self,article,query):
        """ Assign each sentence in the document a score"""
        maxx=0
        maxxx=0
        Query=[]
        Query.append(query)
        headline = article[1]
        sentences = self.split_into_sentences(article[2])
        
        querry=self.remove_smart_quotes(query)
        sentencess=self.split_into_sentences(article[2])
        sentencess.append(querry)
        
        #queryy=self.split_into_sentences(query)
        norm_corpus =self.normalize_corpus(sentences, lemmatize=True)
        norm_corpuss=self.normalize_corpus(sentencess, lemmatize=True)
        
        norm_model_answer =  self.normalize_corpus(Query, lemmatize=True) 
        norm_model_answerquery =  self.normalize_corpus(Query, lemmatize=True) 
        
        vectorizer, corpus_features = self.build_feature_matrix(norm_corpus,feature_type='frequency')
        vectorizerq, query_features = self.build_feature_matrix(norm_corpuss,feature_type='frequency')
        # extract features from model_answer
        model_answer_features = vectorizer.transform(norm_model_answer)
        model_answer_featuresquery = vectorizerq.transform(norm_model_answerquery)
        
        doc_lengths = [len(doc.split()) for doc in norm_corpus]
        doc_lengthss = [len(doc.split()) for doc in norm_corpuss] 
        
        #query_lengths = [len(doc.split()) for doc in norm_query]  
        avg_dl = np.average(doc_lengths) 
        avg_qr = np.average(doc_lengthss)
        
        corpus_term_idfs = self.compute_corpus_term_idfs(corpus_features, norm_corpus)
        corpus_term_idfsquery = self.compute_corpus_term_idfs(query_features, norm_corpuss)
        
        for index, doc in enumerate(Query):
    
            doc_features = model_answer_features[index]
            #doc_featuress = model_answer_featuresquery[index]
            self.bm25_scores = self.compute_bm25_similarity(doc_features,corpus_features,doc_lengths,avg_dl,corpus_term_idfs,k1=1.5, b=0.75)         
            print(' self.bm25_scores')
            print(self.bm25_scores/maxx)
            maxxx=max(self.bm25_scores)
            self.semantic_similarity_scores=[]
     
        for indexx, doc in enumerate(Query):

                doc_featuress = model_answer_featuresquery[indexx]
                self.bm25_scoresquery = self.compute_bm25_similarityqr(doc_featuress,query_features,doc_lengthss,avg_qr,corpus_term_idfsquery,k1=1.5, b=0.75)
            
                maxx=max(self.bm25_scoresquery)
        for i, s in enumerate(sentences):
                score1=self.sentence_similarity(s,query)
                score2=self.sentence_similarity(query,s)
                if score1 is not None and score2 is not None:
                    score=(score1+score2)/2
                    self.semantic_similarity_scores.append(score)
                elif score1 is not None and score2 is None:
                    self.semantic_similarity_scores.append(score1)
                elif score2 is not None and score1 is None:
                    self.semantic_similarity_scores.append(score2)
        print('self.semantic_similarity_scores')
        print(self.semantic_similarity_scores)
        doc_index=0
        sim_score=[]
        sim_scorecos=[]
        for score_tuple in zip(self.semantic_similarity_scores,self.bm25_scores):
            sim_scorecos.append((score_tuple[1]/maxxx))
        print('bm25')
        print( sim_scorecos)
        for score_tuple in zip(self.semantic_similarity_scores,self.bm25_scores):
            sim_score.append((score_tuple[0]+(score_tuple[1]/maxxx))/2)
        print('sim_score')
        print(sim_score)
        for tuple_ in zip(sentences,sim_score):
            s=tuple_[0]
            self._scores[s]=tuple_[1]
        print('self._scores[s]')
        print(self._scores[s])
        
        
                    
              
    def generate_summaries(self,query):
        
                self.dict_ = {'task':[],'paper_id':[],'title':[],'summary': [],'score':[],'sentences':[]}
                jj=1
                ii=1
          
                #tasks=['what is the immune system response to 2019-ncov ?'
                   
                 #  ]
           
    
                #for query in tasks:
                for article in self._articles:
                    self._scores = Counter()
                    self.score(article,query)
                    highest_scoring = self._scores.most_common(SUMMARY_LENGTH)
                    print('highest_scoring')
                    print(highest_scoring)
                    totalsentences = self.split_into_sentences(article[2])
                    summarylist=[]
                    summr=[sent[0] for sent in highest_scoring]
                   
                    for sentence in totalsentences:
                        for sumsen in  summr:
                            if sentence==sumsen:
                                summarylist.append(sentence)        
                    # Appends highest scoring "representative" sentences, returns as a single summary paragraph.
                    summary=' '.join([sent for sent in summarylist])
                    s=0
                    for scr in highest_scoring:
                        s=s+scr[1]
                    s=s/12    
                    
                    
                    ''' 
                    print('**task**')
                    print(query)
                    print(" **Title: **")
                    print(article[1])
                    print(highest_scoring)
                    print('**summary**')
                    print(summary)
                    print('____________________________________________________________________________________')
                    '''   
                    
                    self.dict_['sentences'].append(highest_scoring)
                    self.dict_['summary'].append(summary)
                    self.dict_['title'].append(article[1])
                    self.dict_['paper_id'].append(article[0])
                    self.dict_['task'].append(query)
                    self.dict_['score'].append(s)
                self.papers = pd.DataFrame(self.dict_, columns=['task','paper_id','title','summary','score','sentences'])
                return self.papers
       

    def remove_smart_quotes(self, text):
       
       
        #text=re.sub("([\(\[].*?[\)\]][\(\[].*?[\)\]]+)", ' ', text)
        text=re.sub("[\(\[].*?[\)\]]", '', text)
        
        try:
            url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', text)
            repl_url = url.group(3)
            text = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)',repl_url, text)
        except:
            pass #there might be emails with no url in them
        #text=re.sub("[\[]()*?[\]]", "", text)#remove in-text citation
        
        text=re.sub(r"[<>()(,)|&©ø\[\]\'\";?~*!]", ' ', text) #remove <>()|&©ø"',;?~*!
        text=re.sub("(\\t)", ' ', text) #remove escape charecters
        text=re.sub("(\\r)", ' ', text) 
        text=re.sub("(\\n)", ' ', text)
        text= re.sub("(\s+)",' ',text) #remove multiple space
        
        text = re.sub(r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}', 'MAIL', text)
    # remove doi
        text = re.sub(r'https\:\/\/doi\.org[^\s]+', 'DOI', text)
    # remove https
        text = re.sub(r'(\()?\s?http(s)?\:\/\/[^\)]+(\))?', '\g<1>LINK\g<3>', text)
    # remove single characters repeated at least 3 times for spacing error (e.g. s u m m a r y)
        text = re.sub(r'(\w\s+){3,}', ' ', text)
    # replace tags (e.g. [3] [4] [5]) with whitespace
        text = re.sub(r'(\[\d+\]\,?\s?){3,}(\.|\,)?', ' \g<2>', text)
    # replace tags (e.g. [3, 4, 5]) with whitespace
        text = re.sub(r'\[[\d\,\s]+\]', ' ',text)
     # replace tags (e.g. (NUM1) repeated at least 3 times with whitespace
        text = re.sub(r'(\(\d+\)\s){3,}', ' ',text)
    # replace '1.3' with '1,3' (we need it for split later)
        text = re.sub(r'(\d+)\.(\d+)', '\g<1>,\g<2>', text)
    # remove all full stops as abbreviations (e.g. i.e. cit. and so on)
        text = re.sub(r'\.(\s)?([^A-Z\s])', ' \g<1>\g<2>', text)
    # correctly spacing the tokens
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        text=re.sub(r"[<>()(,)|&©ø\[\]\'\";?~*!]", ' ', text) #remove <>()|&©ø"',;?~*!
        text=re.sub("(\\t)", ' ', text) #remove escape charecters
        text=re.sub("(\\r)", ' ', text) 
        text=re.sub("(\\n)", ' ', text)
        text= re.sub("(\s+)",' ',text) #remove multiple space
        text=re.sub("doi", ' ',text)
        text=re.sub("bioRxiv", ' ',text)
        text=re.sub("author", ' ',text)
        text=re.sub("authors", ' ',text)
        text=re.sub("authors", ' ',text)
        text=re.sub("All rights reserved", ' ',text)
        text=re.sub("preprint", ' ',text)
    # return lowercase text
        return text.lower()
        
        
        
      


    def split_into_sentences(self, text):
        new=[]
        tok = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tok.tokenize(self.remove_smart_quotes(text))
        sentences = [sent.replace('\n', '') for sent in sentences if len(sent) > 20]   
        words=['author','authors','permissions','doi','medRxiv',' preprint','copyright', 'holder']
        for sentt in sentences :
            sent=word_tokenize(sentt)
            
            if (('bioRxiv'not in sent) and ('author'not in sent) and ('authors' not in sent) and('permission'not in sent) and('permissions'not in sent)and('doi'not in sent)
                     and ('medrxiv'not in sent) and ('medRxiv'not in sent)and(' Java'not in sent)and('java'not in sent) and(' javascript'not in sent)and(' JavaScript'not in sent)and(' preprint'not in sent)and('JavaScript'not in sent)and('copyright'not in sent) and ('holder'not in sent)):  
                    
                       new.append(sentt)
       
        
                 

        return new
    def lemmatize_text(self,text):
    
        pos_tagged_text = self.pos_tag_text(text)
        lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                             else word                     
                             for word, pos_tag in pos_tagged_text]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text

    def remove_special_characters(self,text):
        tokens = self.tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def remove_stopwords(self,text):
        stopword_list = nltk.corpus.stopwords.words('english')
        tokens = self.tokenize_text(text)
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text

    
  
    def build_feature_matrix(self,documents, feature_type='frequency',
                         ngram_range=(1, 1), min_df=0.0, max_df=1.0):

        feature_type = feature_type.lower().strip()  
    
        if feature_type == 'binary':
            vectorizer = CountVectorizer(binary=True, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
        elif feature_type == 'frequency':
            vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
        elif feature_type == 'tfidf':
            vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, 
                                     ngram_range=ngram_range)
        else:
            raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

        feature_matrix = vectorizer.fit_transform(documents).astype(float)
    
        return vectorizer, feature_matrix

    def compute_corpus_term_idfs(self,corpus_features, norm_corpus):
    
        dfs = np.diff(sp.csc_matrix(corpus_features, copy=True).indptr)
        dfs = 1 + dfs # to smoothen idf later
        total_docs = 1 + len(norm_corpus)
        idfs = 1.0 + np.log(float(total_docs) / dfs)
        return idfs
    def compute_bm25_similarity(self,doc_features, corpus_features,
                            corpus_doc_lengths, avg_doc_length,
                            term_idfs, k1=1.5, b=0.75):
    # get corpus bag of words features
        corpus_features = corpus_features.toarray()
    # convert query document features to binary features
    # this is to keep a note of which terms exist per document
        doc_features = doc_features.toarray()[0]
        doc_features[doc_features >= 1] = 1
    
    # compute the document idf scores for present terms
        doc_idfs = doc_features * term_idfs
    # compute numerator expression in BM25 equation
        numerator_coeff = corpus_features * (k1 + 1)
        numerator = np.multiply(doc_idfs, numerator_coeff)
    # compute denominator expression in BM25 equation
        denominator_coeff =  k1 * (1 - b + 
                                (b * (corpus_doc_lengths / 
                                        avg_doc_length)))
        denominator_coeff = np.vstack(denominator_coeff)
        denominator = corpus_features + denominator_coeff
    # compute the BM25 score combining the above equations
        bm25_scores = np.sum(np.divide(numerator,
                                   denominator),
                         axis=1) 
    
        return bm25_scores
    def tokenize_text(self,text):
        tokens = nltk.word_tokenize(text) 
        tokens = [token.strip() for token in tokens]
        return tokens
    def compute_bm25_similarityqr(self,doc_features, corpus_features,
                            corpus_doc_lengths, avg_doc_length,
                            term_idfs, k1=1.5, b=0.75):
    # get corpus bag of words features
        corpus_features = corpus_features.toarray()
    # convert query document features to binary features
    # this is to keep a note of which terms exist per document
        doc_features = doc_features.toarray()[0]
        doc_features[doc_features >= 1] = 1
    
    # compute the document idf scores for present terms
        doc_idfs = doc_features * term_idfs
    # compute numerator expression in BM25 equation
        numerator_coeff = corpus_features * (k1 + 1)
        numerator = np.multiply(doc_idfs, numerator_coeff)
    # compute denominator expression in BM25 equation
        denominator_coeff =  k1 * (1 - b + 
                                (b * (corpus_doc_lengths / 
                                        avg_doc_length)))
        denominator_coeff = np.vstack(denominator_coeff)
        denominator = corpus_features + denominator_coeff
    # compute the BM25 score combining the above equations
        bm25_scores = np.sum(np.divide(numerator,
                                   denominator),
                         axis=1) 
    
        return bm25_scores

    



def remove_stopwords(sen):     
    sen_new = " ".join([i for i in sen if i not in stop_words])          
    return sen_new

def add(request):
    import pandas as pd
    import csv
    val1 = request.GET['query']
    val2=int(request.GET['num'])
    bm25_index = RankBM25Index(data)
    results = None
    added = []
    #for3 word in keywords:
    #print(word)
    #print("word_result")
    word_result = bm25_index.search(val1).results
    results = word_result
    dc = results.sort_values(by='Score', ascending=False)

    dc.reset_index(drop=True, inplace=True)
    dc.to_csv('indexx.csv', index=False)
    
    csv.field_size_limit(100000000)
    results=pd.DataFrame()
    ff= open("indexx.csv", encoding="utf-8-sig")
    reader = csv.reader(ff, delimiter=',')
    next(reader)
    summaries= Summarizer(reader)
    results=summaries.generate_summaries(val1)
    resultss = results.sort_values(by='score', ascending=False)
    resultss.head(50)
    resultss.to_csv('results.csv', index=False)
    df=pd.read_csv('results.csv')
    sentences = [] 
    for s in df['summary']: 
        sentences.append(sent_tokenize(s))

    sentences = [y for x in sentences for y in x]



    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ") 
        # make alphabets lowecase 
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
   
    # Extract word vectors 
    word_embeddings = {} 
    f = open(r"C:\meriem\Mémoire\glove6b\glove.6B.100d.txt", encoding='utf-8') 
    for line in f: 
        values = line.split() 
        word = values[0] 
        coefs = np.asarray(values[1:], dtype='float32')    
        word_embeddings[word] = coefs 
    f.close()
    sentence_vectors = [] 
    for i in clean_sentences: 
        if len(i) != 0: 
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001) 
        else: 
            v = np.zeros((100,)) 
        sentence_vectors.append(v)

    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)): 
        for j in range(len(sentences)): 
            if i != j: 
                 sim_mat[i][j] = cosine_similarity (sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
 
    import networkx as nx 
    nx_graph = nx.from_numpy_array(sim_mat) 
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in 
                           enumerate(sentences)), reverse=True)

    summary=[]
    for i in range(val2):
        summary.append((ranked_sentences[i][1]))
    

    summm='\n'.join(map(str, summary))
    
   

    
    context={}
    context["content"]=summm
    context["nub"]=val2
    
    return render(request, 'web/result.html',context)

# Create your views here.




def index(request):
    program=Programming.objects.all()
    d={'program':program}
    return render(request, 'web/home.html', d)



# AJAX
def load_courses(request):
   programming_id = request.GET.get('programming')
   courses = Course.objects.filter(programming_id=programming_id).order_by('name')
   return render(request, 'web/city_dropdown_list_options.html', {'courses': courses})

