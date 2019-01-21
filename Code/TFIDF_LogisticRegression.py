import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict,train_test_split,GridSearchCV
from scipy.sparse import hstack
from sklearn.pipeline import make_union
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_auc_score

import re

train = pd.read_csv(r"C:\Users\Admin\Desktop\Y2S1\CS3244\train_clean.csv")
test = pd.read_csv(r"C:\Users\Admin\Desktop\Y2S1\CS3244\test_clean.csv")

vect_word = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word',
                        stop_words= 'english',ngram_range=(1,3),dtype=np.float32,max_df = 0.9)

tr_vect = vect_word.fit_transform(train['comment_text'].values.astype('U'))

ts_vect = vect_word.transform(test['comment_text'].values.astype('U'))

#print(ts_vect)
X = tr_vect
x_test = ts_vect 

target_col = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
y = train[target_col]

prd = np.zeros((x_test.shape[0],y.shape[1]))
cv_score =[]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# toxic = 0.001

score=[]
for i,col in enumerate(target_col):
    lr = LogisticRegression(C=2,random_state = i,class_weight = 'balanced')
    print('Building {} model for column:{''}'.format(i,col))
    lr.fit(X,y[col])
    #pred =  lr.predict(X_test)
    #print('\nConfusion matrix\n',confusion_matrix(y_test[col],pred))
    #score.append(roc_auc(y[col],pred))
    prd[:,i] = lr.predict_proba(x_test)[:,1]

#print(sum(score)/6)

for col in target_col:
	print("Column:",col)
	pred =  lr.predict(X)
	#print(r2_score(y[col], pred))
	#print('\nConfusion matrix\n',confusion_matrix(y[col],pred))
	#print(pd.crosstab(y[col], pred, rownames=['True'], colnames=['Predicted'], margins=True))
	print(classification_report(y[col],pred))
    
for col in target_col:
	print("Column:",col)
	pred_pro = lr.predict_proba(X)[:,1]
	frp,trp,thres = roc_curve(y[col],pred_pro)
	print(auc(frp,trp))


prd_1 = pd.DataFrame(prd,columns=y.columns)
submit = pd.concat([test['id'],prd_1],axis=1)
#submit.to_csv(r"C:\Users\Admin\Desktop\Y2S1\CS3244\submission_4.csv",index=False)
#submit.head()