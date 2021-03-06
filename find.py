import gensim, logging
import sys
import pprint
#import _uniout
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
_file=sys.argv[1]
# train word2vec on the two sentences
model = gensim.models.Word2Vec.load(_file)

#r=model.similarity(sys.argv[2],sys.argv[3] )
r=model.most_similar(positive=[sys.argv[2], sys.argv[3]], negative=[sys.argv[4]])
#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(r)
for key,v in r:
    print(unicode(key.strip(), "utf8", errors="ignore"))

