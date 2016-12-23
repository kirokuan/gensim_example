# import modules & set up logging
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import pprint
import pandas as pd
import sys
import os.path as path
import fnmatch
import os


pp= pprint.PrettyPrinter(indent=4)
_file=sys.argv[1]
root='/tmp2/yckuan/data/filter2/'
batch=10000
for file in os.listdir(root):
    if fnmatch.fnmatch(file,_file):
	print root,file
    	df = pd.read_csv(root+file,header=0)#open(input_file, 'rb').read()
    	x=[]
    	for index, row in df.iterrows():
	    try:
		d=df.loc[index,'text'].split(' ')
    	    	x.append(d)
            except:
		print "ex"
pp.pprint('read done')	
size=int(sys.argv[2])
model = gensim.models.Word2Vec(x, min_count=1, size=size)# import modules & set up logging
model.save('../model/word2vec.'+_file.replace('*','')+'.'+str(size)+'.model')
pp.pprint('done')



