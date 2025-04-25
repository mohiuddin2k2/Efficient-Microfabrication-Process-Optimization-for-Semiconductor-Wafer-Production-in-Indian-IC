from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import pandas as pd
import numpy as np
from keras.optimizers import Adam
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import os
import pickle

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,multilabel_confusion_matrix
from sklearn import metrics

main = tkinter.Tk()
main.title("semiconductor") 
main.geometry("1000x650")

global filename
global x_train,y_train,x_test,y_test
global X, Y
global le
global dataset
accuracy = []
precision = []
recall = []
fscore = []
global classifier
global cnn_model

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset)+"\n\n")

def preprocessDataset():
    global X, y
    global le
    global dataset
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)
    print(dataset.info())
    text.insert(END,str(dataset.head())+"\n\n")
    
    # Create a count plot
    sns.set(style="darkgrid")  # Set the style of the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    # Replace 'dataset' with your actual DataFrame and 'Drug' with the column name
    ax = sns.countplot(x='Label', data=dataset, palette="Set3")
    plt.title("Count Plot")  # Add a title to the plot
    plt.xlabel("Categories")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
    # Annotate each bar with its count value
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

    plt.show()  # Display the plot
    
    dataset.fillna(0,inplace=True)
    X=dataset.drop('Label',axis=1)
    y=dataset['Label']  
    
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X,y= smote.fit_resample(X, y)
    text.insert(END,"Total records found in dataset: "+str(X.shape[0])+"\n\n")
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=52)
    text.insert(END,"Total records found in dataset to train: "+str(x_train.shape[0])+"\n\n")
    text.insert(END,"Total records found in dataset to test: "+str(x_test.shape[0])+"\n\n")
    print(x_train)
def analysis():
    # Create a count plot
    sns.set(style="darkgrid")  # Set the style of the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    # Replace 'dataset' with your actual DataFrame and 'Drug' with the column name
    ax = sns.countplot(x=y, palette="Set3")
    plt.title("Count Plot")  # Add a title to the plot
    plt.xlabel("Categories")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
    # Annotate each bar with its count value
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
    plt.show()
    

def classifier():
    global x_train, y_train
   
    clf = MLPClassifier()
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END, " Extra Tree Classifier Precision : " + str(p) + "\n")
    text.insert(END, "Extra Tree Classifier Recall    : " + str(r) + "\n")
    text.insert(END, "Extra Tree Classifier FMeasure  : " + str(f) + "\n")
    text.insert(END, "Extra Tree Classifier Accuracy  : " + str(a) + "\n\n")
    # Compute confusion matrix
    cm = confusion_matrix(y_test,predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Extra Tree Classifier Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    # Compute classification report
    report = classification_report(y_test,predict)
    # Display confusion matrix in the Text widget
    text.insert(END, "Confusion Matrix:\n")
    text.insert(END, str(cm) + "\n\n")
    # Display classification report in the Text widget
    text.insert(END, "Classification Report:\n")
    text.insert(END, report)
   

def RandomForestclassifier():
    global x_train, y_train, x_test, y_test
    global rfc
    
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    
    predict = rfc.predict(x_test)
    
    p = precision_score(y_test, predict, average='macro', zero_division=0) * 100
    r = recall_score(y_test, predict, average='macro', zero_division=0) * 100
    f = f1_score(y_test, predict, average='macro', zero_division=0) * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    # Display precision, recall, F1-score, and accuracy in the Text widget
    text.insert(END, "Random Forest Classifier Precision: " + str(p) + "\n")
    text.insert(END, "Random Forest Classifier Recall: " + str(r) + "\n")
    text.insert(END, "Random Forest Classifier FMeasure: " + str(f) + "\n")
    text.insert(END, "Random Forest Classifier Accuracy: " + str(a) + "\n\n")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, predict)
    
    # Compute classification report
    report = classification_report(y_test, predict)
    
    # Display confusion matrix in the Text widget
    text.insert(END, "Confusion Matrix:\n")
    text.insert(END, str(cm) + "\n\n")
    
    # Display classification report in the Text widget
    text.insert(END, "Classification Report:\n")
    text.insert(END, report)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Classifier Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    report= classification_report(y_test, predict)
    print(report)

def Prediction():
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, f'{filename} Loaded\n')
    test = pd.read_csv(filename)
    test.dropna(inplace=True)
    for i in range(len(test)):
        input_data = test.iloc[i, :].values.reshape(1, -1)
        predict = rfc.predict(input_data) 
        text.insert(END, f'Input data for row {i}: {input_data}\n')
        if predict == 0:
            predicted_data = "Fault"
        elif predict == 1:
            predicted_data = "NO Fault"
        text.insert(END, f'Predicted output for row {i}: {predicted_data}\n')

def graph():
    # Create a DataFrame
    df = pd.DataFrame([
    ['MLP', 'Precision', precision[0]],
    ['MLP', 'Recall', recall[0]],
    ['MLP', 'F1 Score', fscore[0]],
    ['MLP', 'Accuracy', accuracy[0]],
    ['RFC', 'Precision', precision[-1]],
    ['RFC', 'Recall', recall[-1]],
    ['RFC', 'F1 Score', fscore[-1]],
    ['RFC', 'Accuracy', accuracy[-1]],
    ], columns=['Parameters', 'Algorithms', 'Value'])

    # Pivot the DataFrame and plot the graph
    pivot_df = df.pivot_table(index='Parameters', columns='Algorithms', values='Value', aggfunc='first')
    pivot_df.plot(kind='bar')
    # Set graph properties
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # Display the graph
    plt.show()
def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text="Semiconductor")
title.config(bg='white', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=200,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=500,y=100)
preprocessButton.config(font=font1)

analysisButton = Button(main, text="Applying SMOTE", command=analysis)
analysisButton.place(x=200,y=150)
analysisButton.config(font=font1) 

knnButton = Button(main, text=" ExtraTreesClassifier", command=classifier)
knnButton.place(x=500,y=150)
knnButton.config(font=font1)

LRButton = Button(main, text="RandomForestclassifier", command=RandomForestclassifier)
LRButton.place(x=200,y=200)
LRButton.config(font=font1)

predictButton = Button(main, text="Prediction", command=Prediction)
predictButton.place(x=500,y=200)
predictButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=200,y=250)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=500,y=250)
exitButton.config(font=font1)

                            
font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1) 

#main.config(bg='orange')
main.mainloop()
