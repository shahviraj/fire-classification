
# coding: utf-8

# In[1]:


from __future__ import print_function
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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.decomposition import MiniBatchDictionaryLearning
from time import time


# In[3]:


#Extract data
comb_data = h5py.File('combustion_img_13.mat','r')


# In[4]:


X_train = comb_data['train_set_x'][()]
y_train = comb_data['train_set_y'][()]
X_test = comb_data['test_set_x'][()]
y_test = comb_data['test_set_y'][()]


# In[5]:


X_train_final = np.zeros([54000,1250])
n=len(X_train_final[:,1])
np.shape(y_train)


# In[6]:


#Feature extraction using Histogram of Gradients
#Training data
for i in range(0,n):
    temp_image = X_train[:,i]
    
    temp_image = np.reshape(temp_image,[250,100])
    temp_image = temp_image.T
    temp_fd = hog(temp_image, orientations=5, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1))
    
    X_train_final[i,:] = temp_fd


# In[9]:


#Extract features of test data using Histogram of Gradients
X_test_final = np.zeros([18000,1250])
b=len(X_test_final[:,1])
for i in range(0,b):
    temp_image = X_test[:,i]
    
    temp_image = np.reshape(temp_image,[250,100])
    temp_image = temp_image.T
    temp_fd = hog(temp_image, orientations=5, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1))
    
    X_test_final[i,:] = temp_fd


# In[8]:


np.shape(X_train)


# In[10]:


#Feature extraction using Dictionary Learning
#Training data
print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=10, alpha=1, n_iter=100)
X_train_dict = dico.fit_transform(X_train.T)
np.shape(X_train_dict)
dt = time() - t0
print('done in %.2fs.' % dt)


# In[11]:


#Express test data in terms of (Dictionary) learned features
X_test_dict = dico.transform(X_test.T)
np.shape(X_test_dict)


# In[15]:


#Random forest classifier for HOG features # n_features = 1250
num_features = "auto" #default option "auto" = sqrt(n_features); "log2" = log2(n_features); None = n_features 
clf = RandomForestClassifier(n_estimators=2, max_features=num_features, max_depth=4, random_state=0)
clf.fit(X_train_final, np.ravel(y_train))

#Random forest classifier for dictionary features # n_features = 10
num_features_d = "auto" #default option "auto" = sqrt(n_features); "log2" = log2(n_features); None = n_features 
clfd = RandomForestClassifier(n_estimators=2, max_features=num_features_d, max_depth=4, random_state=0)
clfd.fit(X_train_dict, np.ravel(y_train))


# In[20]:


#Test classification accuracy using Random Forest Classifier for HOG features

y_test_predict=clf.predict(X_test_final)
acc = accuracy_score(y_test, y_test_predict)
print('Accuracy of Random Forest Classifier using HOG features is...',acc)



# In[21]:


#Test classification accuracy using Random Forest Classifier for dictionary features

y_test_predictd=clfd.predict(X_test_dict)
acc_d = accuracy_score(y_test, y_test_predictd)
print('Accuracy of Random Forest Classifier using Dictionary learned features is...',acc_d)


#### CODE IS COMPLETE TILL THIS POINT ####


# In[ ]:


#### CODE NEEDS TO BE FIXED BEYOND THIS POINT ####

#KFold Cross validation

kf = KFold(n_splits=3)
kf.get_n_splits(X_train_final)
print(kf)


# In[ ]:


for train_index, test_index in kf.split(X_train_final):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train_kfold, X_test_kfold = X_train_final[train_index], X_train_final[test_index]
    y_train_kfold, y_test_kfold = y_train[train_index], y_train[test_index]

np.shape(y_train)

clf = svm.SVC(kernel='linear', C=1).fit(X_train_kfold, y_train_kfold)
clf.score(X_test_kfold, y_test_kfold)


# In[ ]:


#Grid search
X_train_gs, X_test_gs, y_train_gs, y_test_gs = train_test_split(
    X_train_final, y_train, test_size=0.3, random_state=0)


# In[ ]:


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

