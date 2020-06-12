#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pickle
import pandas as pd


# In[7]:


data=pd.read_csv('Spam SMS Collection',sep='\t',names=['label','message'])


# In[8]:


data.head()


# In[9]:


#import all the necessary libraries for dataset
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[10]:


#cleaning the message
corpus=[]
ps=PorterStemmer()


# In[12]:


for word in range(0,data.shape[0]):
    #removing the special character 
    message=re.sub(pattern='[^a-zA-z]',repl=' ',string=data.message[word])
    #converting the message to lower case
    message=message.lower()
    #splitting the words in message
    words=message.split()
    #removing stop words
    words=[word for word in words if word not in set(stopwords.words('english'))]
    #stemming the words
    words=[ps.stem(word) for word in words ]
    #joining the stemmed words
    message=' '.join(words)
    #building corpus of model
    corpus.append(message)


# In[25]:


#creating the bag of model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()


# In[26]:


#extracting dependent variable 
y=pd.get_dummies(data['label'],drop_first=True)


# In[27]:


y.head()


# In[28]:


#creating a pickle file for count vectorizer
pickle.dump(cv,open('cv-transform.pkl', 'wb'))


# In[29]:


#creating model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)


# In[30]:


#fitting naive bayes to model
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB(alpha=0.3)
classifier.fit(X_train,y_train)


# In[31]:


# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'spam-sms-mnb-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[ ]:




