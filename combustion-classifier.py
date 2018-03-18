
# coding: utf-8

# In[12]:


from __future__ import print_function
import numpy as np
import scipy as sc
import cv2 as cv2
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
from skimage.feature import hog
from skimage import data, exposure
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[13]:


#Extract data
#comb_data = h5py.File('../Aditya_data/combustion_img_13.mat','r')
comb_data = h5py.File('combustion_img_13.mat','r')


# In[14]:


X_train = comb_data['train_set_x'][()]
y_train = comb_data['train_set_y'][()]
y_train = np.ravel(y_train)
X_test = comb_data['test_set_x'][()]
y_test = comb_data['test_set_y'][()]
y_test = np.ravel(y_test)
X_valid = comb_data['valid_set_x'][()]
y_valid = comb_data['valid_set_y'][()]
y_valid = np.ravel(y_valid)


# In[15]:


X_train_final = np.zeros([54000,1250])
n=len(X_train_final[:,1])
np.shape(y_train)


# In[16]:


#Feature extraction using Histogram of Gradients
#Training data
for i in range(0,n):
    temp_image = X_train[:,i]
    
    temp_image = np.reshape(temp_image,[250,100])
    temp_image = temp_image.T
    temp_fd = hog(temp_image, orientations=5, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1))
    
    X_train_final[i,:] = temp_fd


# In[17]:


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


# In[18]:


#Extract features of validation data using Histogram of Gradients
X_valid_final = np.zeros([9000,1250])
b=len(X_valid_final[:,1])
for i in range(0,b):
    temp_image = X_valid[:,i]
    
    temp_image = np.reshape(temp_image,[250,100])
    temp_image = temp_image.T
    temp_fd = hog(temp_image, orientations=5, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1))
    
    X_valid_final[i,:] = temp_fd


# In[19]:


np.shape(X_train)


# In[20]:


#Feature extraction using Dictionary Learning
#Training data
print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=10, alpha=1, n_iter=100)
X_train_dict = dico.fit_transform(X_train.T)
np.shape(X_train_dict)
dt = time() - t0
print('done in %.2fs.' % dt)


# In[21]:


#Express test data in terms of (Dictionary) learned features
X_test_dict = dico.transform(X_test.T)
np.shape(X_test_dict)


# In[22]:


#Express validation data in terms of (Dictionary) learned features
X_valid_dict = dico.transform(X_valid.T)
np.shape(X_valid_dict)


# In[23]:


#Random forest classifier for HOG features # n_features = 1250
num_features = "auto" #default option "auto" = sqrt(n_features); "log2" = log2(n_features); None = n_features 
clf = RandomForestClassifier(n_estimators=2, max_features=32, max_depth=2, random_state=0)
clf.fit(X_train_final, y_train)

#Random forest classifier for dictionary features # n_features = 10
num_features_d = "auto" #default option "auto" = sqrt(n_features); "log2" = log2(n_features); None = n_features 
clfd = RandomForestClassifier(n_estimators=2, max_features=num_features_d, max_depth=4, random_state=0)
clfd.fit(X_train_dict, y_train)


# In[24]:


#Random forest classifier for HOG features # n_features = 1250
num_features = "auto" #default option "auto" = sqrt(n_features); "log2" = log2(n_features); None = n_features 
clf = RandomForestClassifier(n_estimators=6, max_features=1250, max_depth=4, random_state=0)
clf.fit(X_train_final, y_train)


# In[25]:


#Test classification accuracy using Random Forest Classifier for HOG features

y_test_predict=clf.predict(X_test_final)
acc = accuracy_score(y_test, y_test_predict)
print('Accuracy of Random Forest Classifier using HOG features is...',acc)



# In[26]:


#Random forest classifier for dictionary features # n_features = 10
num_features_d = "auto" #default option "auto" = sqrt(n_features); "log2" = log2(n_features); None = n_features 
clfd = RandomForestClassifier(n_estimators=6, max_features=10, max_depth=4, random_state=0)
clfd.fit(X_train_dict, y_train)


# In[27]:


#Test classification accuracy using Random Forest Classifier for dictionary features

y_test_predictd=clfd.predict(X_test_dict)
acc_d = accuracy_score(y_test, y_test_predictd)
print('Accuracy of Random Forest Classifier using Dictionary learned features is...',acc_d)


# In[28]:


##Compare the performance to Decision Tree, Extra Trees, AdaBoost, Voting and Gradient Tree Boosting.

#Decision Tree
clf_dt = tree.DecisionTreeClassifier(max_depth=4, max_features=30)
clf_dt.fit(X_train_final, y_train)
y_test_predict_dectree=clf_dt.predict(X_test_final)
acc_dt = accuracy_score(y_test, y_test_predict_dectree)
print('Accuracy of Decision Tree Classifier using HOG features is...',acc_dt)
clf_dt_d = tree.DecisionTreeClassifier(max_depth=4, max_features=9)
clf_dt_d.fit(X_train_dict, y_train)
y_test_predict_dectree_d=clf_dt_d.predict(X_test_dict)
acc_dt_d = accuracy_score(y_test, y_test_predict_dectree_d)
print('Accuracy of Decision Tree Classifier using Dictionary learned features is...',acc_dt_d)


# In[29]:


#Extra Trees
extclf= ExtraTreesClassifier(n_estimators=250, max_depth=4, max_features=30, random_state=0)
extclf.fit(X_train_final, y_train)
y_test_predict_exttree=extclf.predict(X_test_final)
acc_et = accuracy_score(y_test, y_test_predict_exttree)
print('Accuracy of Extra Trees Classifier using HOG features is...',acc_et)
extclf_d= ExtraTreesClassifier(n_estimators=10, max_depth=4, max_features=9, random_state=0)
extclf_d.fit(X_train_dict, y_train)
y_test_predict_exttree_d=extclf_d.predict(X_test_dict)
acc_et_d = accuracy_score(y_test, y_test_predict_exttree_d)
print('Accuracy of Extra Trees Classifier using Dictionary learned features is...',acc_et_d)


# In[30]:


#Gradient Boosting Classifier
clfgb = GradientBoostingClassifier(n_estimators=30, learning_rate=1.0, max_depth=4, max_features=3, random_state=0)
clfgb.fit(X_train_final, y_train)
y_test_predict_gradboost=clfgb.predict(X_test_final)
acc_gb = accuracy_score(y_test, y_test_predict_gradboost)
print('Accuracy of Gradient Boosting Classifier using HOG features is...',acc_gb)
clfgb_d = GradientBoostingClassifier(n_estimators=9, learning_rate=1.0, max_depth=4, max_features=3, random_state=0)
clfgb_d.fit(X_train_dict, y_train)
y_test_predict_gradboostd=clfgb_d.predict(X_test_dict)
acc_gb_d = accuracy_score(y_test, y_test_predict_gradboostd)
print('Accuracy of Gradient Boosting Classifier using Dictionary learned features is...',acc_gb_d)


# In[31]:


#AdaBoost Classifier
clfab = AdaBoostClassifier()
clfab.fit(X_train_final, y_train)
y_test_predict_adboost=clfab.predict(X_test_final)
acc_ab = accuracy_score(y_test, y_test_predict_adboost)
print('Accuracy of AdaBoost Classifier using HOG features is...',acc_ab)
clfab_d = AdaBoostClassifier()
clfab_d.fit(X_train_dict, y_train)
y_test_predict_adboostd=clfab_d.predict(X_test_dict)
acc_ab_d = accuracy_score(y_test, y_test_predict_adboostd)
print('Accuracy of AdaBoost Classifier using Dictionary learned features is...',acc_ab_d)


# In[32]:


#Voting Classifier
clfrf = RandomForestClassifier(random_state=1)
clfvote = VotingClassifier(estimators=[('rf', clfrf)], voting='hard')
clfvote.fit(X_train_final, y_train)
y_test_predict_vote=clfvote.predict(X_test_final)
acc_v = accuracy_score(y_test, y_test_predict_vote)
print('Accuracy of Voting Classifier using HOG features is...',acc_v)
clfrfd = RandomForestClassifier(random_state=1)
clfvoted = VotingClassifier(estimators=[('rf', clfrfd)], voting='hard')
clfvoted.fit(X_train_dict, y_train)
y_test_predict_voted=clfvoted.predict(X_test_dict)
acc_vd = accuracy_score(y_test, y_test_predict_voted)
print('Accuracy of Voting Classifier using Dictionary learned features is...',acc_vd)


# In[33]:


#KFold Cross validation

print('Merging original training and validation data for HOG features...')
X_full = np.append(X_train_final,X_valid_final,axis=0)

print('Merging original training and validation data for Dictionary learned features...')
X_full_d = np.append(X_train_dict,X_valid_dict,axis=0)

y_full = np.append(y_train,y_valid)

kf = KFold(n_splits=2)  #n_splits can be increased, but will result in higher running time


# In[34]:


np.shape(X_full_d)


# In[ ]:


#Grid search on K=2 fold cross validation dataset
for train_index, test_index in kf.split(X_train_final):
    
    #Generate new cross-validation dataset
    print("TRAIN:", train_index, "TEST:", test_index)
    #reduced dataset with HOG features
    X_train_gs, X_test_gs = X_full[train_index], X_full[test_index]
    #reduced dataset with Dictionary learned features
    X_train_gs_d, X_test_gs_d = X_full_d[train_index], X_full_d[test_index]    
    y_train_gs, y_test_gs = y_full[train_index], y_full[test_index]
    
    #Get optimal hyper-parameters for this CV dataset, using HOG features
    
    tuned_parameters_rf = [{"n_estimators": [2, 3], "max_depth": [2, 4], "max_features": [10, 30]}] #add max_features also
    tuned_parameters_dt = [{"max_depth": [2, 4], "max_features": [10, 30]}] 
    tuned_parameters_et = [{"max_depth": [2, 4], "max_features": [10, 30]}]
    tuned_parameters_gb = [{"n_estimators": [2, 3], "max_depth": [2, 4], "max_features": [10, 30]}]
    tuned_parameters_ab = [{"n_estimators": [2, 3], }]
    tuned_parameters_vc = [{"estimators": [2, 3]}]
    
    clf_considered = RandomForestClassifier()
    #clf_considered = tree.DecisionTreeClassifier()
    #clf_considered = ExtraTreesClassifier()
    #clf_considered = GradientBoostingClassifier()
    #clf_considered = AdaBoostClassifier()
    #clf_considered = VotingClassifier
   
   
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf_gs = GridSearchCV(clf_considered,# this handle can be changed for any other classifier
                              tuned_parameters_rf,  scoring='%s_macro' % score)
        clf_gs.fit(X_train_gs, np.ravel(y_train_gs))

        #HOG features
        print("Best parameters set found on development set, using HOG features:")
        print()
        print(clf_gs.best_params_)
        print()
        print("Grid scores on development set, using HOG features:")
        print()
        means = clf_gs.cv_results_['mean_test_score']
        stds = clf_gs.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf_gs.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set, using HOG features.")
        print("The scores are computed on the full evaluation set, using HOG features.")
        print()
        y_true, y_pred = y_test_gs, clf_gs.predict(X_test_gs)
        print(classification_report(y_true, y_pred))
        print()
    
        
        
    #Get optimal hyper-parameters for this CV dataset using Dictionary learned features
    tuned_parameters_rf_dl = [{"n_estimators": [2, 3], "max_depth": [2, 4], "max_features": [4, 9]}] #add max_features also
    tuned_parameters_dt_dl = [{"max_depth": [2, 4], "max_features": [4, 9]}] 
    tuned_parameters_et_dl = [{"max_depth": [2, 4], "max_features": [4, 9]}]
    tuned_parameters_gb_dl = [{"n_estimators": [2, 3], "max_depth": [2, 4], "max_features": [4, 9]}]
    tuned_parameters_ab_dl = [{"n_estimators": [2, 3]}]
    tuned_parameters_vc_dl = [{"estimators": [2, 3]}]
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

      
        #Dictionary learned features
        clf_gs_d = GridSearchCV(clf_considered,# this handle can be changed for any other classifier
                              tuned_parameters_rf_dl,  scoring='%s_macro' % score)
        clf_gs_d.fit(X_train_gs_d, y_train_gs)
        print("Best parameters set found on development set, using Dictionary learned features:")
        print()
        print(clf_gs_d.best_params_)
        print()
        print("Grid scores on development set, using Dictionary learned features:")
        print()
        means = clf_gs_d.cv_results_['mean_test_score']
        stds = clf_gs_d.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf_gs_d.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set, using Dictionary learned features.")
        print("The scores are computed on the full evaluation set, using Dictionary learned features.")
        print()
        y_true_d, y_pred_d = y_test_gs, clf_gs_d.predict(X_test_gs_d)
        print(classification_report(y_true_d, y_pred_d))
        print()
            

