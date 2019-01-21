import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc

import re

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

vect_word = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='words',
                        stop_words= 'english',ngram_range=(1,3),dtype=np.float32,max_df=0.9)
vect_char = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='char',
                        stop_words= 'english',ngram_range=(1,4),dtype=np.float32,max_df=0.9)


tr_vect = vect_word.fit_transform(train['comment_text'].astype('U'))
ts_vect = vect_word.transform(test['comment_text'].astype('U'))


X = tr_vect
x_test = ts_vect

target_col = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
y = train[target_col]

prd = np.zeros((x_test.shape[0],y.shape[1]))
cv_score =[]

for i,col in enumerate(target_col):
    gnb = MultinomialNB()
    print('Building {} model for column:{''}'.format(i,col))
    gnb.fit(X,y[col])
    prd[:,i] = (gnb.predict(x_test))

for col in target_col:
	print("Column:",col)
	pred =  gnb.predict(X)
	#print('\nConfusion matrix\n',confusion_matrix(y[col],pred))
	print(pd.crosstab(y[col], pred, rownames=['True'], colnames=['Predicted'], margins=True))
	print(classification_report(y[col],pred))

prd_1 = pd.DataFrame(prd,columns=y.columns)
submit = pd.concat([test['id'],prd_1],axis=1)
submit.to_csv('submissionNB.csv',index=False)
submit.head()
