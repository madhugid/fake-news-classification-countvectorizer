#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[35]:


df = pd.read_csv('/home/gsunilmadhusudanreddy/Training/NLP/fake news classification - kaggle/train.csv')
df.head()


# In[36]:


df = df.dropna()


# In[37]:


X = df.drop('label', axis =1)
X.head()


# In[38]:


y = df['label']


# In[39]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


# In[40]:


news = df.copy()
news.reset_index(inplace = True)


# In[41]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[42]:


ps = PorterStemmer()
wordnet = WordNetLemmatizer()


# In[43]:


corpus = []


# In[44]:


for i in range(0, len(news)):
    message = re.sub('[^a-zA-Z]', ' ', news['title'][i])
    message = message.lower()
    message = message.split()
    message = [ps.stem(word) for word in message if not word in stopwords.words('english')]
    message = ' '.join(message)
    corpus.append(message) 


# In[45]:


from sklearn.feature_extraction.text import CountVectorizer


# In[46]:


cv = CountVectorizer(max_features = 5000,ngram_range = (1,3))

X = cv.fit_transform(corpus).toarray()


# In[47]:


y = news['label']


# In[48]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


# In[49]:


cv.get_feature_names()[:10]


# In[50]:


cv.get_params()


# In[51]:


count_df = pd.DataFrame(X_train,columns = cv.get_feature_names())


# In[52]:


count_df.head()


# In[53]:


import matplotlib.pyplot as plt


# In[54]:


def plot_confusion_matrix(cm, classes, normalize = False, title = "confusion matrix", cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[55]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


# In[56]:


from sklearn import metrics
import numpy as np
import itertools


# In[57]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[58]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score


# In[59]:


from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(max_iter = 50)


# In[60]:


linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])


# Multinomial Classifier with Hyperparameter

# In[61]:


classifier=MultinomialNB(alpha=0.1)


# In[62]:


previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))


# In[63]:


feature_names = cv.get_feature_names()


# In[66]:


classifier.fit(X_train,y_train)


# In[72]:


classifier.coef_


# In[73]:


sorted(zip(classifier.coef_[0], feature_names), reverse=True)[:20]


# HashingVectorizer

# In[74]:


hs_vectorizer=HashingVectorizer(n_features=5000,non_negative=True)
X=hs_vectorizer.fit_transform(corpus).toarray()


# In[75]:


X.shape


# In[76]:


X


# In[77]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[78]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[ ]:




