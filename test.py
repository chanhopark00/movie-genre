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
import ast 
from gensim import models
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
import codecs
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

def split_data(data, proportion):
    n = data.shape[0]
    tmp = 0
    x_test = [] 
    y_test = []
    x_train = [] 
    y_train = []
    train_all = []
    for index, row in data.iterrows():
        if n* proportion > tmp:
            x_test.append(row['w_description'])
            y_test.append(row['v_genre'])
        else:
            x_train.append(row['w_description'])
            y_train.append(row['v_genre'])
            train_all.append(row['all'])
        tmp+= 1
    x_test = array(x_test)
    y_test = array(y_test)
    x_train = array(x_train)
    y_train = array(y_train)

    return x_test, y_test , x_train, y_train,train_all

def num_hidden_layer1(num_input, num_output, size):
    # alpha is usually set to be a value to be between 2-10
    alpha = 10
    return [int(size /(num_input + num_output) / alpha)]
def num_hidden_layer2(num_input, num_output,size):
    # when single hidden layer
    return [int(2*((num_output +2)*num_input)**0.5)]
def num_hidden_layer3(num_input, num_output,size):
    # when two hidden layer
    return [int(((num_output+2)*num_input)**0.5 + 2*(num_input/(num_output+2))**0.5), int(num_output*(num_input/(num_output+2))**0.5)]

def wordvec(data,model):
    i = 0.0
    out = 0.0
    for word in str(data).split():
        if word in model.vocab:
            out += model[word]
            i += 1
    if i != 0:
        out = out/i
    else:
        print(data)
    return out

def label2vector(genre,allg):
    out = [0]*len(allg)
    dic = ast.literal_eval(genre)  
    for g in dic:
        out[allg.index(g['name'])] = 1
        break
    return out

def tag(data,lemma,stop_words):
    filtered = ""
    try:
        data = word_tokenize(data)
        data = nltk.pos_tag(data)
        for word in data:
            if word[0] not in stop_words and len(word[0])>2:
                try: 
                    word[1].index('V')
                    filtered = filtered + " " + lemma.lemmatize(word[0].lower(),'v')
                except:
                    filtered = filtered + " " + lemma.lemmatize(word[0].lower())
    except:
        pass  
    return filtered
def add(row):
    if len(str(row['tagline'])) > 3:
        return str(row['overview'])+" "+str(row['tagline'])
    else:
        return str(row['overview'])
def tolist(data):
    dic = ast.literal_eval(data)  
    out = []
    for g in dic:
        out.append(g['name'])
    return out
def tovec(row,allg):
    out = [0]*20
    for d in row:
        out[allg.index(d)] = 1
    return out

def in_max(l):
    m = 0
    out = 0
    for i,j,w in zip(range(20),l,weights):
        if m < j*w/2:
            m = j*w/2
            out = i
    return out
def score(true,final):
    correct = 0
    for i,j in zip(true,final):
        if i == j:
            correct +=1
    print(correct/len(true))

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(20)
    plt.xticks(tick_marks, allg[:20], rotation=45)
    plt.yticks(tick_marks, allg[:20])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    

file_dir = "./data/movies_metadata.csv"
data = pd.read_csv(file_dir)
data = data[['genres','overview','tagline']]
allg = []
for index,row in data['genres'].items():
    dic = ast.literal_eval(row)  
    for g in dic:
        try:
            allg.index(g['name'])
        except:
            allg.append(g['name'])

data['all'] = data['genres'].apply(tolist)
data['v_genre'] = data['genres'].apply(label2vector, args=(allg,))
print(data.head(5))
count_g = [0]* len(allg)
for i,row in data['v_genre'].items():
    for j,r in zip(range(len(row)),row):
        if r == 1:
            count_g[j] +=1
rm_list = [ 'Carousel Productions', 'Vision View Entertainment', 'Telescene Film Group Productions','Aniplex', 'GoHands', 'BROSTA TV', 'Mardock Scramble Production Committee', 'Sentai Filmworks', 'Odyssey Media', 'Pulser Productions', 'Rogue State', 'The Cartel']
rm_index = []
for i,row in data['v_genre'].items():
    for j,r in zip(range(len(row)),row):
        if r == 1 and j > 19 and j <32:
            rm_index.append(i)

data = data.drop(rm_index)
data['v_genre'] = data['v_genre'].apply(lambda x: x[:20])
#concatenate overview and tagline
data['description'] = data[['overview','tagline']].apply(add, axis =1)
stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
data['description'] = data['description'].apply(tag,args=(lemma,stop_words,))
no_genre = []
for i,v in data['v_genre'].items():
    if v == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
        no_genre.append(i)
data = data.drop(no_genre)
print("lemma over word2vec")
data = data.drop(['genres','tagline'], axis=1)
model2= models.KeyedVectors.load_word2vec_format('./models/GoogleNews.bin',binary=True)
data['w_description'] = data['description'].apply(wordvec, args=(model2,))

rm = []
for i,v in data['w_description'].items():
    try:
        if v == 0.0:
            rm.append(i) 
    except:
        pass
data = data.drop(rm,axis=0)

print("word2vec over")
data['all'] = data['all'].apply(tovec,args=(allg,))
x_test, y_test , x_train, y_train, train_all= split_data(data,0.1)

vec_size = 300
in_len = vec_size # number of input feature
out_len = 20 # number of output label
hidden_layer = num_hidden_layer2(in_len,out_len,data.shape[0])
activation = 'relu'
ep = 80
print(hidden_layer)

dist = [0] * 20
y_train_int = []
# convert vector label to int label for SMOTE
for y in y_train:
    for i,g in zip(range(20),y):
        if g == 1:
            dist[i] += 1
            y_train_int.append(i)
weights = compute_class_weight('balanced', range(20), y_train_int)
print(weights)

weight_dic = {}
for i,w in enumerate(weights):
    weight_dic[i] = w
print("Class weight is : \n",weight_dic)

nn_weight = Sequential()
if len(hidden_layer) == 1:
    nn_weight.add(Dense(hidden_layer[0],input_dim=in_len,activation=activation))
elif len(hidden_layer) == 2:
    nn_weight.add(Dense(hidden_layer[0],input_dim=in_len,activation=activation))
    nn_weight.add(Dense(hidden_layer[1],activation=activation))
nn_weight.add(Dense(out_len, activation='softmax'))
nn_weight.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history_class = nn_weight.fit(x_train, y_train, epochs=100, batch_size=20, verbose=1, validation_data=(x_test, y_test), class_weight=weight_dic)

nn = Sequential()
if len(hidden_layer) == 1:
    nn.add(Dense(hidden_layer[0],input_dim=in_len,activation=activation))
elif len(hidden_layer) == 2:
    nn.add(Dense(hidden_layer[0],input_dim=in_len,activation=activation))
    nn.add(Dense(hidden_layer[1],activation=activation))
elif len(hidden_layer) == 3:
    nn.add(Dense(hidden_layer[0],input_dim=in_len,activation=activation))
    nn.add(Dense(hidden_layer[1],activation=activation))
    nn.add(Dense(hidden_layer[2],activation=activation))
nn.add(Dense(out_len, activation='softmax'))
nn.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = nn.fit(x_train, y_train, epochs=ep, batch_size=20, verbose=1, validation_data=(x_test, y_test))


pred_weight = nn_weight.predict(x_test)
pred_nn = nn.predict(x_test)
final = []
for r1,r2 in zip(pred_weight,pred_nn):
    add = []
    for v1,v2 in zip(r1,r2):
        add.append((v1+v2)/2)
    final.append(in_max(add))
final = array(final)
true = []
for i in train_all:
    true.append(in_max(i))
score(true,final)

title ="Confusion matrix"
cm = confusion_matrix(true,final)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12,10))
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')



