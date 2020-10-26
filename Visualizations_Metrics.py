#!/usr/bin/env python
# coding: utf-8

# ## Visualizations to add

# In[ ]:


import boto3

# Convert your existing model to JSON
saved_model = model.to_json()

# Write JSON object to S3 as "keras-model.json"
client = boto3.client('s3')
client.put_object(Body=saved_model,
                  Bucket='BUCKET_NAME',
                  Key='cnn-model.json')


# In[ ]:


from keras.models import model_from_json

# Read the downloaded JSON file
with open('cnn-model.json', 'r') as model_file:
    loaded_model = model_file.read()

# Convert back to Keras model
model = model_from_json(loaded_model)

# Confirmation
model.summary()


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True) ### this is all we need to do


# In[ ]:


# plot for training/val loss

from matplotlib import pyplot as plt
plt.clf()
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'y', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


# plot for training/val accuracy

plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b+', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Calculate the confusion matrix
cm = multilabel_confusion_matrix(y_true=y_test, y_pred=y_pred)
disp = plot_confusion_matrix(model, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues)
plt.show()


# In[ ]:


FP = cm.sum(axis=0) - np.diag(cm) 
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)


# In[ ]:


print('Precision: %.3f' % precision_score(y_test, y_pred))


# In[ ]:


print('Recall: %.3f' % recall_score(y_test, y_pred))


# In[ ]:


print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


# In[ ]:


print('F1 Score: %.3f' % f1_score(y_true, y_pred, average='weighted'))


# In[ ]:


from wordcloud import WordCloud,STOPWORDS

plt.figure(figsize=(40,25))
text = df['clean_desc'].values
cloud = WordCloud(stopwords=STOPWORDS,
                    background_color='white',
                    collocations=False,
                    width=2500,
                    height=1800
                    ).generate(" ".join(text))
plt.axis('off')
plt.title("Common Words in Line Descriptions",fontsize=100)
plt.imshow(cloud)


# In[ ]:


from wordcloud import WordCloud,STOPWORDS

plt.figure(figsize=(40,25))
text = df['clean_title'].values
cloud = WordCloud(stopwords=STOPWORDS,
                    background_color='white',
                    collocations=False,
                    width=2500,
                    height=1800
                    ).generate(" ".join(text))
plt.axis('off')
plt.title("Common Words in Line Descriptions",fontsize=100)
plt.imshow(cloud)

