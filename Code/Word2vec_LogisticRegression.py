
import numpy as np
import pandas as pd
import re, string
import pickle
import math

from gensim.models import word2vec
import gensim
import json

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict,train_test_split,GridSearchCV

#with open('toxic_text_vector.train.pkl', 'rb') as f:
#    train_w2v = pickle.load(f)

#print(train_w2v)

train = pd.read_csv(r"C:\Users\Admin\Desktop\Y2S1\CS3244\train_clean.csv") 
test = pd.read_csv(r"C:\Users\Admin\Desktop\Y2S1\CS3244\test_clean.csv")

train.dropna(inplace = True)
test.dropna(inplace = True)

train.comment_text = train.comment_text.apply(lambda x : x.lower())
train.comment_text= train.comment_text.apply(lambda s: re.sub(r'[^\w\s]','',s))
train.comment_text = train.comment_text.apply(lambda s : re.sub("\d+", "", s))
test.comment_text= test.comment_text.apply(lambda x : x.lower())
test.comment_text = test.comment_text.apply(lambda s: re.sub(r'[^\w\s]','',s))
test.comment_text= test.comment_text.apply(lambda s : re.sub("\d+", "", s))


#sentences from train_comments
train_comments = train['comment_text']
test_comments = test['comment_text']


vocab_train = []
vocab_test = []
#print (len(train_comments))
#print (len(test_comments))
for sentences in train_comments:
  vocab_train.append(sentences.split())
for sentences in test_comments:
  vocab_test.append(sentences.split())

vocab = vocab_train + vocab_test

 #initialises and trains model
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words
# Initialize and train the model (this will take some time)
w2v = word2vec.Word2Vec(vocab, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)
w2v.init_sims(replace=True)
#w2v = gensim.models.Word2Vec(vocab, size = 100)



#vocab_dict = {}
#for word, vocab_obj in w2v.wv.vocab.items():
#    temp = w2v[word]
#    vocab_dict[word] = sum(temp)/len(temp)
    
#def text_to_words(raw_text, remove_stopwords=False):
    # 1. Remove non-letters, but including numbers
#    letters_only = re.sub("[^0-9a-zA-Z]", " ", raw_text)
    # 2. Convert to lower case, split into individual words
#    words = letters_only.lower().split()
#    if remove_stopwords:
#        stops = set(stopwords.words("english")) # In Python, searching a set is much faster than searching
#        meaningful_words = [w for w in words if not w in stops] # Remove stop words
#        words = meaningful_words
#    return words 

#sentences_train = train['comment_text'].apply(text_to_words, remove_stopwords=False)
#sentences_test = test['comment_text'].apply(text_to_words, remove_stopwords=False)
    
def makeFeatureVec(words, model, num_features):
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
    # Divide the result by the number of words to get the average
    if nwords == 0:
        nwords = 1
    featureVec = np.divide(featureVec, nwords)
    return featureVec



def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    counter = 0
    # Loop through the reviews
    for review in reviews:
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs

#train.comment_text = train.comment_text.apply(lambda s : s.split())
#test.comment_text = test.comment_text.apply(lambda s : s.split())

X = getAvgFeatureVecs(train["comment_text"], w2v, 300)
x_test = getAvgFeatureVecs(test["comment_text"], w2v, 300)


# we have to train 6 different models with 6 different Y labels

target_col = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
y = train[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

prd = np.zeros((x_test.shape[0],y.shape[1]))

for i,col in enumerate(target_col):
    lr = LogisticRegression(C=2,random_state = i,class_weight = 'balanced')
    print('Building {} model for column:{''}'.format(i,col))
    lr.fit(X,y[col])
    prd[:,i] = lr.predict_proba(x_test)[:,1]
    
#for i,col in enumerate(target_col):
#    gnb = MultinomialNB()
#    print('Building {} model for column:{''}'.format(i,col))
#    gnb.fit(X,y[col])
#   prd[:,i] = (gnb.predict(x_test))
    
for col in target_col:
    print("Column:",col)
    pred = lr.predict(X)
    print('\nConfusion matrix\n',confusion_matrix(y[col],pred))
    
#	pred =  gnb.predict(X)
	#print('\nConfusion matrix\n',confusion_matrix(y[col],pred))
#	print(pd.crosstab(y[col], pred, rownames=['True'], colnames=['Predicted'], margins=True))
#	print(classification_report(y[col],pred))
    
prd_1 = pd.DataFrame(prd,columns=y.columns)
submit = pd.concat([test['id'],prd_1],axis=1)
submit.to_csv(r"C:\Users\Admin\Desktop\Y2S1\CS3244\submission_w2v_7.csv",index=False)
submit.head()