import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import numpy as np 

data_dict = pickle.load(open('./data.pickle','rb'))

data = np.asarray(data_dict['data']) #getting features of images
labels = np.asarray(data_dict['labels']) #getting labels of images

#shuffle is true so that the model does not learn patterns based on order of input data
#stratify = labels for preserving the ratio of classes
X_train,X_test,Y_train,Y_test = train_test_split(data,labels,test_size=0.2,shuffle=True,stratify=labels)

model = RandomForestClassifier()

model.fit(X_train,Y_train)

Y_predict = model.predict(X_test)

score = accuracy_score(Y_predict,Y_test)

print('{}% of samples were classified correctly'.format(score*100))

#saving the model
f = open('model.p','wb')
pickle.dump({'model':model},f)
f.close()


