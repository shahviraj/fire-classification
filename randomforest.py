
# coding: utf-8

# In[92]:


import numpy as np
import scipy as sc
import cv2 as cv2
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
from skimage.feature import hog
from skimage import data, exposure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import svm
from __future__ import print_function
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


# In[49]:


comb_data = h5py.File('C:\\Vignesh\\ME592X\\Assignment\\Assignment 3\\combustion_img_13.mat','r')


# In[116]:


X_train = comb_data['train_set_x'][()]
y_train = comb_data['train_set_y'][()]
X_test = comb_data['test_set_x'][()]
y_test = comb_data['test_set_y'][()]


# In[117]:


X_train_final = np.zeros([54000,1250])
n=len(X_train_final[:,1])
np.shape(y_train)


# In[78]:


#def hog(data):
#    X = np.zeros((len(data[1,:],1250))
#    for i in range(0,len(data[1,:])):
#        temp=data[:,i]
#        temp=np.reshape(temp_image,[250,100])
#        temp=temp.T
#        temp=hog(temp, orientations=5, pixels_per_cell=(10, 10),
#                    cells_per_block=(1, 1))
#        X[i,:] = temp_fd
#    return X[i,:]
for i in range(0,n):
    temp_image = X_train[:,i]
    
    temp_image = np.reshape(temp_image,[250,100])
    temp_image = temp_image.T
    temp_fd = hog(temp_image, orientations=5, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1))
    
    X_train_final[i,:] = temp_fd
    


# In[79]:


X_test_final = np.zeros([18000,1250])
b=len(X_test_final[:,1])
for i in range(0,b):
    temp_image = X_test[:,i]
    
    temp_image = np.reshape(temp_image,[250,100])
    temp_image = temp_image.T
    temp_fd = hog(temp_image, orientations=5, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1))
    
    X_test_final[i,:] = temp_fd


# In[95]:


clf = RandomForestClassifier(n_estimators=2,max_depth=2, random_state=0)
clf.fit(X_train_final, y_train)


# In[96]:


y_test_predict=clf.predict(X_test_final)
accuracy_score(y_test, y_test_predict)


# In[88]:


#KFold Cross validation
kf = KFold(n_splits=3)
kf.get_n_splits(X_train_final)
print(kf)


# In[89]:


for train_index, test_index in kf.split(X_train_final):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train_kfold, X_test_kfold = X_train_final[train_index], X_train_final[test_index]
    y_train_kfold, y_test_kfold = y_train[train_index], y_train[test_index]


# In[ ]:


np.shape(y_train)


# In[90]:


clf = svm.SVC(kernel='linear', C=1).fit(X_train_kfold, y_train_kfold)
clf.score(X_test_kfold, y_test_kfold)


# In[106]:


#Grid search
X_train_gs, X_test_gs, y_train_gs, y_test_gs = train_test_split(
    X_train_final, y_train, test_size=0.3, random_state=0)


# In[107]:


tuned_parameters = [{"classifier__n_estimators": [1, 2, 3, 4, 5], "classifier__max_depth": [2, 4, 6, 8, 10]}]
scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train_gs, y_train_gs)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test_gs, clf.predict(X_test_gs)
    print(classification_report(y_true, y_pred))
    print()


# In[ ]:


from sklearn import svm
svc = svm.SVC()
X_folds = np.array_split(X_train_final, 2)
y_folds = np.array_split(y_train, 2)
scores = list()
for k in range(3):

    X_train = list(X_folds)
    X_test  = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test  = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)


# In[114]:


np.shape(y_train)

