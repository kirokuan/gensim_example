# coding=utf-8
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
logging.basicConfig(filename='lsa/'+sys.argv[1]+'.1000.log',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def fit_lda(X, vocab, num_topics=50, passes=1):
    """ Fit LDA from a scipy CSR matrix (X). """
    print 'fitting lda...'
    return LsiModel( gensim.matutils.Sparse2Corpus(X, documents_columns=False)
	, num_topics=num_topics, chunksize=10000
         ,id2word=vocab)


def print_topics(lda, vocab, n=100):
    """ Print the top words for each topic. """
    topics = lda.show_topics(num_words=n,log=True)
    for ti, topic in enumerate(topics):
        print 'topic %d: ' % (ti)
   	#print(topic.encode('utf-8'))
	print '%s |  %s' % (topic[0],topic[1])


_file=sys.argv[1]
root ='/tmp2/yckuan/data/new/'
texts=[]
for file in os.listdir(root):
    
    if fnmatch.fnmatch(file,_file):
        print file
	df = pd.read_csv(root+file,header=0)#open(input_file, 'rb').read()
        for index, row in df.iterrows():
            try:
		print index
                print row.text
                if isinstance(row.text, basestring):
		    texts.append(row.text)
            except Exception,e:
                print "ex"
		exc_type, exc_value, exc_traceback = sys.exc_info()
		print('exception: '+ str(e))

vec = CountVectorizer(min_df=10)
X = vec.fit_transform(texts)
vocab  = dict((v, k) for k, v in vec.vocabulary_.iteritems())
#pp.pprint(vocab)
lda = fit_lda(X, vocab)
#pp.pprint(lda)
print_topics(lda, vocab)
#print(dict)
