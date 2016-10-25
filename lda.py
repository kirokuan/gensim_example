import logging, gensim, bz2
import pandas as pd
from gensim import corpora
from gensim import matutils
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
import pprint
import os
import os.path as path
import fnmatch
import sys
from sklearn.feature_extraction.text import CountVectorizer

pp= pprint.PrettyPrinter(indent=4)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def fit_lda(X, vocab, num_topics=5, passes=20):
    """ Fit LDA from a scipy CSR matrix (X). """
    print 'fitting lda...'
    return LdaModel( gensim.matutils.Sparse2Corpus(X, documents_columns=False)
	, num_topics=num_topics,passes=passes
         ,id2word=vocab)


def print_topics(lda, vocab, n=10):
    """ Print the top words for each topic. """
    topics = lda.show_topics(topics=-1, topn=n, formatted=False)
    for ti, topic in enumerate(topics):
        print 'topic %d: %s' % (ti, ' '.join('%s/%.2f' % (t[1], t[0]) for t in topic))


_file=sys.argv[1]
root ='/tmp2/yckuan/data/new/'
texts=[]
for file in os.listdir(root):
    
    if fnmatch.fnmatch(file,_file):
        print file
	df = pd.read_csv(root+file,header=0)#open(input_file, 'rb').read()
        for index, row in df.iterrows():
            try:
                #d=df.loc[index,'text'].split(' ')
                if isinstance(row.text, basestring):
		    texts.append(row.text)
            except:
                print "ex"

vec = CountVectorizer(min_df=10)
X = vec.fit_transform(texts)
vocab  = dict((v, k) for k, v in vec.vocabulary_.iteritems())
#pp.pprint(vocab)
lda = fit_lda(X, vocab)
#pp.pprint(lda)
#print_topics(lda, vocab)
#print(dict)
