#Date: 2021-Jul-17
#Programmer: HYUN WOOK KANG

from nltk.corpus import stopwords

import string



stopwords=set(stopwords.words('english'))

corpus='Today I received a gift from my friend and it made me happy.\
 Yesterday, I was joyful because I got a gift from a friend.'

corpus=corpus.lower()

vocab=sorted([word for word in {word.strip(string.punctuation) 
                                for word in corpus.split() 
                                    if word.strip(string.punctuation) not in stopwords}])
print('====vocab====')
print(vocab)

sentences=[sen.strip() for sen in corpus.split('.') if sen]

trainX=[]
trainY=[]
D=10
import numpy as np

np.random.seed(1)

word2index={}
index2word={}
for i, w in enumerate(vocab):
    word2index.update({w:i})
    index2word.update({i:w})

n_vocab=len(vocab)    
window_size=2
for sen in sentences:
    wordsList=[word.strip(string.punctuation) for word in sen.split() if word not in stopwords]
   
    for i in range(len(wordsList)):
        word=wordsList[i]
        x=np.zeros(n_vocab, dtype=np.uint8)
        x[word2index[word]]=1
        trainX.append(x)
        #context words for input
        context_words=[]
        for j in range(i-window_size, i+window_size+1, 1):
            if j!=i and j>=0 and j<len(wordsList):
                context_word=wordsList[j]
                vec=list(np.zeros(n_vocab, dtype=np.uint8))
                vec[word2index[context_word]]=1
                context_words.append(vec)
        trainY.append(context_words)        
    
#initialise weight vectors with uniform distribution    
w1=np.random.uniform(size=(n_vocab, D))
w2=np.random.uniform(size=(D, n_vocab))

def softmax(u):
    y_pred=np.exp(u-max(u))
    return y_pred/np.sum(y_pred)

def backpropagate(h, x, w1, w2, error_rate):
    dl_dw2=np.outer(h, error_rate)
    dl_dw1=np.outer(x, np.dot(w2, error_rate.T))
    w1=w1-(learning_rate*dl_dw1)
    w2=w2-(learning_rate*dl_dw2)

    return w1, w2

learning_rate=0.01
#feedforward network
epochs=30
for epoch in range(epochs):
    loss=0
    for i in range(len(trainX)):
        w_t=trainX[i]
        h=np.dot(w_t,w1)
        u=np.dot(h, w2)
        y_pred=softmax(u)
        #error rate for all context words
        error_rates=0
        for context_word in trainY[i]:
            error_rates+=y_pred-context_word
            loss+=-u[context_word.index(1)]
         
        w1, w2=backpropagate(h, w_t, w1, w2, error_rates)    
        loss += len(trainY[i]) * np.log(np.sum(np.exp(u)))
    print('loss:', loss)

a=w1[word2index['happy']]
b=w1[word2index['joyful']]

cos_sim=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
print(w1)
print('cos similariry:', cos_sim)
 
        