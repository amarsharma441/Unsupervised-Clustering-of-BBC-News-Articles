import os
import pandas as pd
import numpy as np
import nltk
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import gensim.downloader as api

word2vec = api.load('word2vec-google-news-300')

class Utility:
    

    def __init__(self):
        pass
    
    
    def load_data(self,path):
        return pd.read_csv(path)
    
    
    def nltk_tag_to_wordnet_tag(self,nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None
 

    def lemmatize_sentence(self,sentence):
        WN_lemmatizer = WordNetLemmatizer()
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
        wordnet_tagged = map(lambda x: (x[0], self.nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized_sentence.append(word)
            else:        
                lemmatized_sentence.append(WN_lemmatizer.lemmatize(word, tag))
                
        return " ".join(lemmatized_sentence)
    

        
    def remove_non_eng_words(self,text):
        
        words = set(nltk.corpus.words.words())
        text = str(text)
        text = ' '.join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
        return str(text)
        

    
    def remove_bad_chars(self,text,bad_chars=['#','£','%','@','=','-','+',';',':','!','*','?','$','1','2','3','4','5','6','7','8','9','0','&','>','<']):

        text = str(text)
        text = ''.join(i for i in text if not i in bad_chars) 
        return str(text)
    
    
    def get_TFIDF(self,dataframe_text):
        
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([i for i in dataframe_text[0:len(dataframe_text)]])
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        tfidf = pd.DataFrame(dense, columns=feature_names)
        return tfidf
    
    
    def get_CountVector(self,dataframe_text):
        
        vectorizer = CountVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([i for i in dataframe_text[0:len(dataframe_text)]])
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        CountVector = pd.DataFrame(dense, columns=feature_names)
        return CountVector
    
    def get_embeddings(self,text):
        global word2vec
        embedding_vec = np.zeros(300)
        valid_tokens = 0 
        tokens = nltk.word_tokenize(str(text))
        for t in tokens:
            try:
                embedding_vec += word2vec[t]
                valid_tokens += 1 
            except KeyError:
                continue
        return embedding_vec/valid_tokens
