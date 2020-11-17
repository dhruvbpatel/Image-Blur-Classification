#!/usr/bin/env python
# coding: utf-8

# In[40]:


#import 
import pandas as pd
import os
from scipy.misc import *
from matplotlib.pyplot import imread
# im = imread(image.png)
import cv2
from PIL import *

from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.metrics import accuracy_score , recall_score, confusion_matrix


# In[22]:


dataset=pd.read_excel("./../../CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx") ## read xls


# In[30]:


dataset.head()


# In[32]:


dataset.columns


# In[33]:


dataset["Unnamed: 1"].value_counts()


# In[37]:


dataset[dataset["Unnamed: 1"]==-1]


# In[42]:


## get the variance usig laplacian method
def variance_of_laplacian(image):
   
    return cv2.Laplacian(image, cv2.CV_64F).var()


# In[43]:


# loop over the input images
def blur_notblur(imagePath):  # our classification method
   
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    text = -1
 
    if fm < 40:
        text = 1
    return text


# In[44]:


from scipy.misc import imresize

temp_pred=[]
temp_orig=[]
for index, row in dataset.iterrows():
    img_name=row["MyDigital Blur"] ## get the images names from dataset
    img_path=os.path.join('./../../CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet',img_name)
#     print(img_path[:-1])
    temp_pred.append(blur_notblur(img_path[:-1]))  # predicted result 
    temp_orig.append(row[1])  ## original result

import numpy as np

y_pred=np.stack(temp_pred)
y_true=np.stack(temp_orig)


# In[48]:



accuracy_score(y_true,y_pred)


# In[46]:


confusion_matrix(y_true,y_pred)


# In[47]:


fpr, tpr, threshold=roc_curve(y_true,y_pred)
roc_auc = metrics.auc(fpr, tpr)


# In[11]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




