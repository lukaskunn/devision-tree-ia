import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff

def criterioBank():
    data,meta = arff.loadarff("D:\FEI\PROGRAMMING\Python\ia\FEI-CC7711-ArvoreDecisao-main\\bank.arff")

    salMedio = np.asarray(data['average']).reshape(-1,1)
    idade = np.asarray(data['age']).reshape(-1,1)
    features = np.concatenate((salMedio, idade),axis=1)
    target = data['subscribed']

    #print(features)

    Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

    plt.figure(figsize=(25, 10))
    tree.plot_tree(Arvore,feature_names=['average', 'age'],class_names=['yes', 'no'],
                   filled=True, rounded=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(25, 10))
    metrics.plot_confusion_matrix(Arvore,features,target,display_labels=['yes', 'no'], values_format='d', ax=ax)
    plt.show()

criterioBank()