
#Loading the Dataset

import pandas as pd

sms_messages = pd.read_csv("smsspamcollection\SMSSpamCollection",
                           sep = '\t',names=['Label','Messages'])

# Text Cleaning

import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


ps = PorterStemmer()

corpus = []
    
for i in range(0, len(sms_messages)):
    review = re.sub('[^a-zA-Z]', ' ', sms_messages['Messages'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating a BOW model

from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(max_features=2500) # To keep the most frequent word

X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(sms_messages['Label'])

y = y.iloc[:,1].values

# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

#Predict the model

y_pred=spam_detect_model.predict(X_test)

#Confusion Matric and accuracy

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm = confusion_matrix(y_test, y_pred)

cr = classification_report(y_test, y_pred)

accuary = accuracy_score(y_test, y_pred)



# Accuary Rate = 98%














