# Finding Topics with LDA

This package contains a python implementation of the Latent Dirichlet allocation. The inference technique used is collapsed gibbs sampling as in Griffiths and Steyvers (2004), where

![equation](https://latex.codecogs.com/svg.image?P\left(z_{i}=j&space;\mid&space;\mathbf{z}_{-i},&space;\mathbf{w}\right)&space;\propto&space;\frac{n_{-i,&space;j}^{\left(w_{i}\right)}&plus;\beta}{n_{-i,&space;j}^{(\cdot)}&plus;W&space;\beta}&space;\frac{n_{-i,&space;j}^{\left(d_{i}\right)}&plus;\alpha}{n_{-i,&space;n}^{\left(d_{i}\right)}&plus;T&space;\alpha})

and

![equation](https://latex.codecogs.com/svg.image?\hat{\phi}_{j}^{(w)}=\frac{n_{j}^{(\text&space;{w})}&plus;\beta}{n_{j}^{(\cdot)}&plus;W&space;\beta}&space;)

![equation](https://latex.codecogs.com/svg.image?\hat{\theta}_{j}^{(d)}=\frac{n_{j}^{(d)}&plus;\alpha}{n^{(d)}&plus;T&space;\alpha})

---

The package can be found in TestPyPi, and it can be installed via pip, by: 

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ LDA-STA663
```

---


Let's explore how to use the package:

```python
from lda_src.lda import LDA

# load a dataset of your choice containing the documents 

from sklearn.datasets import fetch_20newsgroups

newsgroups_data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                                        remove=('headers', 'footers', 'quotes'),
                                        return_X_y=True)

# this implementation of the collapsed gibbs sampler can currently handle many documents only when using a limited number
# of iterations. It is thus advisable to use fewer documents with more iterations
n_sample = 1000
sample = newsgroups_data[:n_sample]

# Let's jump into the package!

T = 20  # number of topics
alpha = 1 / T
beta = 1 / T

lda_model = LDA(T, alpha, beta)

# the model initialization cleans the data, includes bigrams, formats the documents in compressed 
# numpy form and it initializes the topic assignemnt for each word as well as 
# the Word-Topic Matrix and the Topic-Document Matrix

lda_model.initialize(sample)

# one can now fit the model according to three different implementations: 
# using C++ (default), using numba , or using cython. 

# the three implementations differ significanlty in running times C++ << numba << cython

niter = 1000

# C++
lda_model.fit(niter, method="cpp")

# numba
lda_model.fit(niter, method="numba")

# cython
lda_model.fit(niter, method="cython")

# one can now explore the results of the fitted model by calling
lda_model.topic_results(topic=1, words=10)

# where topic is the topic number [0, T). It will return a dataframe with the top words for the selected topic.

# the created object, after initialization, it contains a battery of attributes that sllow
# the user to explore the underlying data as well as use the cleaned and processed data as input in
# other commony python packages for topic modelling like gensim or sklearn.

# some of these attribues are:

lda_model.corpus  # corpus in the format used by gensim

lda_model.vocabulary  # a dictionary inlcuding {word_id: word}

# the main output of the collapsed gibbs sampler are the estimates for phi and theta:

lda_model.phi

lda_model.theta

# these estimates can be also used to calculate topic or document similarities.

# it is also fun to visualize the topics with the wonderful pyLDAvis. 
# we can just use some of the attributes of out LDA model as arguments for pyLDAvis. 

import pyLDAvis

vis_data = pyLDAvis.prepare(lda_model.phi, lda_model.theta, lda_model.doc_length, lda_model.vocab, lda_model.term_frequency)

pyLDAvis.display(vis_data)
```
