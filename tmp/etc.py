def top2(prediction):
    out = []
    for pred in prediction:
        copy = [[i,p] for i,p in zip(range(len(pred)),pred)]
        copy = sorted(copy, key=lambda a_entry: a_entry[1]) 
        converted = [0] * 26
        converted[copy[-1][0]] = 1
        converted[copy[-2][0]] = 1
        out.append(converted)
    out = array(out)  
    return out

test_pred = nn.predict(x_test)
test_pred = top2(test_pred)

dist = [0] * 26
for y in test_pred:
    for i,g in zip(range(21),y):
        if g == 1:
            dist[i] += 1
s = 0
for d in dist:
    s += d
for d,g in zip(dist,allg):
    print("%.2f "%(d/s) + g)

###
from gensim.test.utils import common_texts

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
j = len(documents)
for i, d in enumerate(data['description']):
    try:
        tagged_data.append(TaggedDocument(words=word_tokenize(d.lower()), tags=[str(i)]))
    except:
        j+=1
        pass
###
max_epochs = 100
vec_sizes = [50, 100,200]
alpha = 0.025
for vec_size in vec_sizes:
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
    model.save("doc2vec.model_"+str(vec_size))
###
model= Doc2Vec.load("./models/doc2vec.bin")
test_data = word_tokenize("".lower())
v1 = model.infer_vector(test_data)

similar_doc = model.docvecs.most_similar([v1])
print(similar_doc)

###
def top2(prediction):
    out = []
    for pred in prediction:
        copy = [[i,p] for i,p in zip(range(len(pred)),pred)]
        copy = sorted(copy, key=lambda a_entry: a_entry[1]) 
        converted = [0] * 26
        converted[copy[-1][0]] = 1
        converted[copy[-2][0]] = 1
        out.append(converted)
    out = array(out)  
    return out

test_pred = nn.predict(x_test)
test_pred = top2(test_pred)

dist = [0] * 26
for y in test_pred:
    for i,g in zip(range(26),y):
        if g == 1:
            dist[i] += 1
s = 0
for d in dist:
    s += d
for d,g in zip(dist,allg):
    print("%.2f "%(d/s) + g)

    # def split_data(data, proportion):
#     n = data.shape[0]
#     tmp = 0
#     x_test = [] 
#     y_test = []
#     x_train = [] 
#     y_train = []
#     for index, row in data.iterrows():
#         if n* proportion > tmp:
#             x_test.append(row['w_description'])
#             y_test.append(row['v_genre'])
#         else:
#             x_train.append(row['w_description'])
#             y_train.append(row['v_genre'])
#         tmp+= 1
#     return x_test, y_test , x_train, y_train
# x_test, y_test , x_train, y_train = split_data(data,0.1)
