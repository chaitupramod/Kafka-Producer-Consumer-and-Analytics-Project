#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  math
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from time import time
import statistics 
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
import sys
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pickle 
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython
from matplotlib import pyplot
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def analysis():
        
    #data distribution plot for class column

    df = pd.read_csv("spambase_test_new.csv")

    ######## Class Distribution Bar Graph ########
    fig = plt.figure()
    ax = fig.add_subplot()
    classes = ['0','1']
    counts = [len(df[df["spam_or_not_spam"]==0]),len(df[df["spam_or_not_spam"]==1])]
    barlist = ax.bar(classes,counts)
    barlist[1].set_color('r')
    plt.title("Class Distribution - 0: Not Spam and 1: Spam ")
    plt.xlabel("Class")
    plt.ylabel("Number of Records")
    plt.show()


    #------------------ Descriptive Statistics ------------------
    #Descriptive statistics using sklearn for each column (mean,median(50 percentile),Min, Max, Average, Std.Dev, Coeff.Var_%)
    descriptive_df = df.describe().T
    descriptive_df.to_csv("descriptive_statistics.csv")
    descriptive_df = descriptive_df.sort_values(by=list(descriptive_df.columns)[1:], ascending=False)
    
    ######check for null values#####
    print("Cheching for Null Values")
    print(df.isnull().values.any())

    #check for data types in csv
    #print(df.dtypes)

    df = pd.read_csv("spambase_test_new.csv")

    #------------------- inferential -----------------------------


    temp_df = pd.DataFrame()
    temp_df["var1"] = [1 if x>0 else 0 for x in list(df["word_freq_money"])]
    temp_df["var2"] = list(df["spam_or_not_spam"])

    csq1=chi2_contingency(pd.crosstab(temp_df["var2"], temp_df["var1"]))
    print("P-value: ",csq1[1])
    #null hypothesis - money and label are independant
    #alternate - dependant
    #p<0.05 - reject null accept alternate

    temp_df["var3"] = [1 if x>0 else 0 for x in list(df["word_freq_technology"])]


    csq2=chi2_contingency(pd.crosstab(temp_df["var2"], temp_df["var3"]))
    print("P-value: ",csq2[1])

    #-------------------- Correlation plot ------------------------ 
    high_corr = df.corr().unstack().sort_values().drop_duplicates()
    #print(high_corr)
    plt.matshow(df.corr())
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.title("Correlation Plot")
    plt.show()

    #Drop correlated columns ---remove features having very high correlation
    print("Dropping columns with highest cooorelation values")
    df = df.drop(['word_freq_415','word_freq_857'], axis=1)
    #print(df)


    #----------------------- Normalization ---------------------------
    print("Performing Normalization")
    for col in df.columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    df.to_csv("preprocessed_data.csv", index=False)
    
    #----------------- Obtaining a permanent train and test set ------
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    
    df = shuffle(df)
    permanent_train, permanent_test = train_test_split(df, test_size=0.2)
    
    permanent_train_index = permanent_train.index
    permanent_test_index = permanent_test.index



    #-------------------------Random Forest -----------------------
    def Random_Forest_calculate(random_state, criterion, max_depth, max_features, n_estimators,x_train, y_train, flag, x_test=0 , y_test=0):
        
        clf_rf1 = RandomForestClassifier(random_state=random_state ,criterion=criterion, max_depth=max_depth, max_features=max_features, n_estimators=n_estimators)
        clf_rf1.fit(x_train, y_train)
        #print("Model Summary")
        #print(clf_rf1)
        if(flag==1):
            filename = 'random_forest_model.sav'
            pickle.dump(clf_rf1,open(filename,'wb'))
        else:
            y_predict = clf_rf1.predict(x_test)
            accuracy = accuracy_score(y_test,y_predict)*100
            accuracy_list.append(accuracy)

            precision_list, recall_list, fscore_list = determine_metrics(y_test,y_predict)
            return accuracy_list, precision_list, recall_list, fscore_list, support_list

    #------------------------Support Vector Machines ----------------
    def SVM_calculate(C, gamma, x_train, y_train, flag, x_test=0, y_test=0):
        clf_svm = svm.SVC(kernel='rbf', C=C, gamma=gamma)
        clf_svm.fit(x_train, y_train)
        if(flag==1):
            filename = 'svm_model.sav'
            pickle.dump(clf_svm,open(filename,'wb'))
        else:
            y_predict = clf_svm.predict(x_test)
            accuracy = accuracy_score(y_test, y_predict)*100
            accuracy_list.append(accuracy)
            precision_list, recall_list, fscore_list = determine_metrics(y_test,y_predict)
            return accuracy_list, precision_list, recall_list, fscore_list
    #------------------------Neural Network ----------------------
    def nn_calculate(input_dim, loss, optimizer, epochs, batch_size, x_train, y_train, flag, x_test=0, y_test=0):
        model = Sequential()
        model.add(Dense(150, input_dim=55, activation = 'relu'))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        model.fit(x,y, epochs = 100, batch_size = 8)
        print("Saved model to disk")
        
        _,accuracy = model.evaluate(x_train, y_train)
        print("ACCURACY NN",accuracy)    
        y_predict = model.predict(x_test)
        for i in y_predict:
            if i<0.5:
                predicted_y_value = 0
            else:
                predicted_y_value = 1
            predicted_list_y_value.append(predicted_y_value)
        
        accuracy = accuracy_score(y_test, predicted_list_y_value)
        precision_list, recall_list, fscore_list = determine_metrics(y_test,predicted_list_y_value)
        accuracy_list_nn.append(accuracy)
        return(accuracy_list, precision_list, recall_list, fscore_list) 
        '''
        rounded = [round(x_test[0]) for x_test in y_predict]
        predictions = model.predict_classes(x_train)
        '''
    def determine_metrics(y_test,y_predict):
        
        metrics = precision_recall_fscore_support(y_test,y_predict)
        precision = metrics[0]
        recall = metrics[1]
        fscore = metrics[2]
        support = metrics[3]
        
        precision_list.append(statistics.mean(precision))
        recall_list.append(statistics.mean(recall))
        fscore_list.append(statistics.mean(fscore))
        #support_list.append(support)
        return (precision_list, recall_list, fscore_list)


    x = permanent_train.iloc[:,:-1]
    y = permanent_train["spam_or_not_spam"]    

    #------------- Grid search random forest ----------------
    '''
    param_grid = { 
    'n_estimators': [200,450],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
    }
    CV_rfc = GridSearchCV(estimator=clf_rf, param_grid=param_grid, cv= 5)
    CV_rfc.fit(x, y)
    print(CV_rfc.best_params_)
    '''

    #------------- Grid search SVM ------------------
    '''
    param_grid = {'C': [0.1, 1, 10, 100], 
                    'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]} 
    # Make grid search classifier
    clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)

    # Train the classifier
    clf_grid.fit(x, y)

    # clf = grid.best_estimator_()
    print("Best Parameters:\n", clf_grid.best_params_)
    #print("Best Estimators:\n", clf_grid.best_estimator_)
    '''

    accuracy_list = []
    precision_list = []
    recall_list =[]
    fscore_list = []
    support_list = []

    accuracy_list_nn = [] 
    precision_list_nn = []
    recall_list_nn = []
    fscore_list_nn = []

    accuracy_list_svm = []
    precision_list_svm = []
    recall_list_svm =[]
    fscore_list_svm = []
    support_list_svm = []

    #------------------- K-fold Cross Validation-----------------
    '''
    k_fold = KFold(n_splits=4, random_state=None)
    count = 0
    for train_idx, test_idx in k_fold.split(x):
        #print("TRAIN:", train_idx, "TEST:", test_idx)
        predicted_list_y_value = []
        x_train = x.iloc[train_idx]
        x_test  = x.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]
        
        #-----------K-fold in Random Forest ---------------
        accuracy_list, precision_list, recall_list, fscore_list, support_list = Random_Forest_calculate(42 ,'entropy',8,'auto', 450, x_train, y_train, 0, x_test, y_test)
        
        #-----------K-fold in Support Vector Machines-------
        accuracy_list_svm, precision_list_svm, recall_list_svm, fscore_list_svm = SVM_calculate(9, 0.8, x_train, y_train,0, x_test, y_test )
        
        #-----------K-fold in Neural Network ----------
        accuracy_list_nn, precision_list_nn, recall_list_nn, fscore_list_nn = nn_calculate(55, 'binary_crossentropy','adam', 1, 5, x_train, y_train,0, x_test, y_test )
        
        count = count+1
        if count == 4:
            
            average_accuracy = statistics.mean(accuracy_list)
            average_precision = statistics.mean(precision_list)
            average_recall = statistics.mean(recall_list)
            average_fscore_list = statistics.mean(fscore_list)
            print("Avearge accuracy_RF", average_accuracy)
            print("Average_precision_RF", average_precision)
            print("Average_recall_RF",average_recall)
            print("Average F-score", average_fscore_list )
            
            average_accuracy_svm = statistics.mean(accuracy_list_svm)
            average_precision_svm = statistics.mean(precision_list_svm)
            average_recall_svm = statistics.mean(recall_list_svm)
            average_f1score_svm = statistics.mean(fscore_list_svm)
            print("Average accuracy SVM",average_accuracy_svm)
            print("Average_precision_SVM", average_precision_svm)
            print("Average_recall_SVM",average_recall_svm)
            print("Average f1 score SVM", average_f1score_svm)
                

            average_accuracy_nn = statistics.mean(accuracy_list_nn)
            average_precision_nn = statistics.mean(precision_list_nn)
            average_recall_nn = statistics.mean(recall_list_nn)
            average_f1score_nn = statistics.mean(fscore_list_nn)
            print("Average accuracy nn",average_accuracy_nn)
            print("Average_precision_nn", average_precision_nn)
            print("Average_recall_nn",average_recall_nn)
            print("Average f1 score nn", average_f1score_nn)
    '''
    
        
    x_permanent_test = permanent_test.iloc[:,:-1]
    y_permanent_test = permanent_test["spam_or_not_spam"]
    x_permanent_train = permanent_train.iloc[:,0:55]
    y_permanent_train = permanent_train.iloc[:,55]


    #-------------Neural Network on Permanent_train and permanent_test----------

    model = Sequential()
    model.add(Dense(150, input_dim=55, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    #model.fit(x_permanent_train,y_permanent_train, epochs = 100, batch_size = 8)
    model.fit(x_permanent_train,y_permanent_train, epochs = 100, batch_size = 8)

    model.save_weights("model_150_batch_8.h5")
    print("Model Summary")
    print(model.summary())
    

    y_predicted_values = model.predict(x_permanent_test)
    
    predicted_list_y_value_last = []

    for i in y_predicted_values:
            if i<0.5:
                predicted_y_value_last = 0
            else:
                predicted_y_value_last = 1
                    
            predicted_list_y_value_last.append(predicted_y_value_last)
            
    accuracy_NN = accuracy_score(y_permanent_test, predicted_list_y_value_last)
    print("accuracy when NN is used", accuracy_NN*100)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))




    #------------ Random forest on permanent train and test set -----------
    rf_model = Random_Forest_calculate(42 ,'entropy',8,'auto', 450, x_permanent_train , y_permanent_train,1)
    rf_loaded_model = pickle.load(open('random_forest_model.sav', 'rb'))
    rf_y_predicted_values = rf_loaded_model.predict(x_permanent_test)
    accuracy_score_rf = accuracy_score(y_permanent_test, rf_y_predicted_values)
    print("accuracy_score of random forest",accuracy_score_rf*100)

    #train SVM on permanent train set
    svm_model = SVM_calculate(10, 1, x_permanent_train, y_permanent_train, 1)
    svm_loaded_model = pickle.load(open('svm_model.sav', 'rb'))
    svm_y_predicted_values = svm_loaded_model.predict(x_permanent_test)
    accuracy_score_svm = accuracy_score(y_permanent_test, svm_y_predicted_values)
    print("accuracy_score of SVM",accuracy_score_svm*100)

    from sklearn.metrics import classification_report

    print("            Classification Report for Random Forest")
    print(classification_report(y_permanent_test, rf_y_predicted_values))


    print("            Classification Report for Support Vector Machine")
    print(classification_report(y_permanent_test, svm_y_predicted_values))

    print("            Classification Report for Neural Network")
    print(classification_report(y_permanent_test, predicted_list_y_value_last))



    #------------------ t-SNE --------------------------------------


    tsne_df = df.iloc[:,:-1]
    tsne_label = df.iloc[:,-1]

    
    """
    print("Fitting TSNE   ...   ...   ...")
    X_embedded = TSNE(n_components=3).fit_transform(tsne_df)
    print(X_embedded)

    with open('parrot_two.pkl', 'wb') as f:
        pickle.dump(X_embedded, f)
        
    """

    with open('parrot.pkl', 'wb') as f:
        pickle.dump(tsne_label, f)

    file1 = open('parrot.pkl', 'rb')
    file2 = open('parrot_two.pkl', 'rb')

    X_embedded = pickle.load(file2)
    tsne_label = pickle.load(file1)



    color_list = []
    for i in tsne_label:
        if(i==1):
            color_list.append("blue")
        elif(i==0):
            color_list.append("red")



    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(X_embedded[:,0], X_embedded[:,1],X_embedded[:,2],c=color_list) 
    pyplot.show()


    X_train = X_embedded[permanent_train_index]
    Y_train = tsne_label[permanent_train_index]

    X_test = X_embedded[permanent_test_index]
    Y_test = tsne_label[permanent_test_index]

   

    """
    param_grid = {'C': [0.1, 1, 10, 100], 
                    'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]} 
    # Make grid search classifier
    clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)

    # Train the classifier
    clf_grid.fit(X_embedded, tsne_label)

    clf = grid.best_estimator_()
    print("Best Parameters:\n", clf_grid.best_params_)
    #print("Best Estimators:\n", clf_grid.best_estimator_)
    """

    clf_svm_tsne = svm.SVC(decision_function_shape='ovo',C=1,gamma=0.1)
    

    clf_svm_tsne.fit(X_train,Y_train)
    Y_pred = clf_svm_tsne.predict(X_test)
    print("Classification Report for T-SNE + Support Vector Machine approach")
    print(classification_report(Y_test, Y_pred))


    #-------------------------------------------------------------------------------------------------------



    #print("Random Forest predicted values",rf_y_predicted_values)
    #print("SVM Predicted values",svm_y_predicted_values)
    #print("Neural Network",predicted_list_y_value_last)

    rf_y_predicted_values = list(np.reshape(rf_y_predicted_values,-1))
    svm_y_predicted_values = np.reshape(svm_y_predicted_values,-1)
    predicted_list_y_value_last = np.reshape(predicted_list_y_value_last,-1)


    data = list(zip(rf_y_predicted_values,svm_y_predicted_values,predicted_list_y_value_last))

    df_ensemble = pd.DataFrame(data, columns = ['RF', 'SVM','NN'])

    print(df_ensemble)

    df_ensemble["ensemble_predictions"] = ""
    df_ensemble["ensemble_predictions"] = df_ensemble.mode(axis=1)
    
    df_ensemble.to_csv("ensemble_predictions.csv", index=False)

    print("*"*100)
    print("ENSEMBLE PREDICTIONS")
    print(df_ensemble["ensemble_predictions"])
    print(classification_report(y_permanent_test, list(df_ensemble["ensemble_predictions"])))

