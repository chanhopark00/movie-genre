import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def tag(data):
    stop_words = set(stopwords.words('english'))
    filtered = ""
    try:
        data = word_tokenize(data)
        for word in data:
            if word not in stop_words and len(word)>2:
                filtered = filtered + " " + word 
    except:
        pass  
    return filtered

file_dir = "./imdb_dataset.json"
data = pd.read_json(file_dir)
data = data[['title','description']]
data['description'] = data['description'].apply(tag)
tagged_data = []
for i, d in enumerate(data['description']):
    try:
        tagged_data.append(TaggedDocument(words=word_tokenize(d.lower()), tags=[str(i)]))
    except:
        pass
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
model.build_vocab(tagged_data)
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha
model.save("doc2vec.model")
print("Model Saved")
model= Doc2Vec.load("doc2vec.model")
test_data = word_tokenize("I love chatbots".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)
print(model.docvecs['1'])