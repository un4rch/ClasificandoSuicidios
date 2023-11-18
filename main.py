# Visualizacion de embeddings: https://projector.tensorflow.org/
# TODO al guardar los embeddings, verificar que se guardan bien, ya que con to_csv se guardan distinto
# TODO multinomialNB da error con embeddings por contener valores negativos, ver si se puede arreglar y sino, pues no usarlo
# TODO verificar que los valores se predicen bien: https://www.kaggle.com/code/rutujapotdar/suicide-text-classification-nlp#Conclusion

# Algoritmos: 
#import ast
# Warnings
#import warnings
#warnings.filterwarnings('ignore')
# System libraries
import os
import sys
import csv
# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
# Data processing libraries
import pandas as pd # frames (tables)
import numpy as np # linear algebra
#import string
# NLP preprocessing libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from preprocessor import Preprocessor
#from sklearn.feature_extraction.text import TfidfVectorizer
# Data serialization 
import pickle
# Classification algorithms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV # Search best classifier for hyperparameter combinations
from sklearn.feature_selection import SelectKBest,chi2,f_classif
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
# Evaluation metrics
import evaluate
from sklearn import metrics

preprocessedFile = None
unpreprocessedFile = "Suicide_Detection.csv"
guardarPreproceso = "Preprocessed_Suicide_Detection.csv"
output_dir = "output"
train = True
visualization = False
textLengthsFilter = 10000 # Se eliminan los textos cuya longitud sea mas de 10000 caracteres
histogramIntervals = 500
pca_dimensions = 200
doc2vec_vectors_size = 1500
doc2vec_model = "Suicide_Detection_doc2vec.model"
pca_model = "Suicide_Detection_pca.model"
prediction_model = "EnsembleMethods_model.pkl"

def cargarDataset(pFileName):
    # Cargar los datos
    df_tmp = pd.read_csv(pFileName)
    # Coger solo las columnas que nos interesan
    df = pd.DataFrame()
    df['text'] = df_tmp['text']
    df['class'] = df_tmp['class']
    return df

def preprocesado(dataFrame, doc2vec_model, pca_model):
    if df.shape[0] > 1000:
        # Obtener 10000 instancias
        dataFrame = dataFrame.sample(n=10000, random_state=42)
    # Filtrar textos por longitud usando un umbral
    dataFrame = dataFrame[dataFrame['text_length']<=textLengthsFilter]
    # Preprocesamos los datos
    preprocessor = Preprocessor()
    x_prep,y_prep,doc2vec_model,pca_model = preprocessor.doc2vec(dataFrame['text'], dataFrame['class'], pca_dimensions=pca_dimensions, doc2vec_vectors_size=doc2vec_vectors_size, doc2vec_model=doc2vec_model, pca_model=pca_model)
    # Convertir de np.array a tuplas
    x_prep = [tuple(point) for point in x_prep.tolist()]
    # Guardar el preproceso
    if guardarPreproceso != None:
            #dataFrame.to_csv(guardarPreproceso) #¿¿¿por que al guardar asi se guardan distinto???
            # Guardar datos preprocesados
            with open(guardarPreproceso, "w") as file:
                writer = csv.writer(file)
                writer.writerow(["text","class"])
                for idx,point in enumerate(x_prep):
                    writer.writerow([point,y_prep[idx]])
            print(f"    Fichero guardado: {guardarPreproceso}")
            if train:
                # Guardar los modelos de d0c2vec y pca
                doc2vec_model.save(unpreprocessedFile.split(".")[0]+"_doc2vec.model")
                with open(unpreprocessedFile.split(".")[0]+"_pca.model", "wb") as file:
                    pickle.dump(pca_model, file)
                print(f"    Fichero guardado: {unpreprocessedFile.split('.')[0]+'_doc2vec.model'}")
                print(f"    Fichero guardado: {unpreprocessedFile.split('.')[0]+'_pca.model'}")
    # Devolvemos el dataFrame preprocesado
    dataFrame['text'] = x_prep
    dataFrame['class'] = y_prep
    return dataFrame

def histogramTextsLengths(dataFrame, fileName):
    # Create a histogram with intervals of 500
    plt.hist(dataFrame['text_length'], bins=range(0, max(dataFrame['text_length']) + histogramIntervals + 1, histogramIntervals), edgecolor='black')

    # Set labels and title
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Text Lengths')

    # Save the plot as a PNG file
    plt.savefig(f"{output_dir}/{fileName}")
    plt.close()
    print(f"    Fichero guardado: {output_dir}/{fileName}")

def boxplotTextLengths(dataFrame, fileName):
    sns.boxplot(x="text_length", y="class", data=dataFrame)
    plt.savefig(f"{output_dir}/class_separated_{fileName}")
    plt.close()
    print(f"    Fichero guardado: {output_dir}/class_separated_{fileName}")
    sns.boxplot(x="text_length", data=dataFrame)
    plt.savefig(f"{output_dir}/{fileName}")
    plt.close()
    print(f"    Fichero guardado: {output_dir}/{fileName}")

def visualizeBalanced(dataFrame, fileName):
    classCnt = df['class'].value_counts()
    #print(classCnt)

    plt.figure(figsize=((20,5)))

    plt.subplot(1,2,1)
    sns.countplot(df,x='class')

    plt.subplot(1,2,2)
    plt.pie(classCnt,labels = classCnt.index,autopct='%.0f%%')

    plt.savefig(f"{output_dir}/{fileName}")
    plt.close()
    print(f"    Fichero guardado: {output_dir}/{fileName}")

def saveConfussionMatrix(confussion_matrix, dType, fileName, cmap):
    plt.figure()
    sns.heatmap(confussion_matrix, annot=True, fmt=dType, cmap=cmap)
    plt.title(fileName.split(".")[0])
    #plt.ylabel("Clusters")
    #plt.xlabel("Clases")
    plt.savefig(f"{output_dir}/{fileName}", format='png')
    plt.close()
    print(f"    Fichero guardado: {output_dir}/{fileName}")

if __name__ == "__main__":
    # Cargar dataset sin preprocesar
    df = cargarDataset(unpreprocessedFile)
    # Calcular una columna que contenga las longitudes de los textos (sirve para el preprocesado)
    df['text_length'] = df['text'].apply(len)
    # Eliminar si ha quedado algun valor vacio al cargar el dataset
    df.dropna(inplace=True)
    # Crear directorio donde se van a guardar las graficas y ficheros generados
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if visualization:
        print("[*] Generando graficas de visualizacion de los datos...")
        # Visualizar si los datos estan balanceados
        visualizeBalanced(df, "check_balanced.png")
        # Generar un histograma que muestra en numero de instancias que tienen cierta cantidad de letras por intervalos
        histogramTextsLengths(df, "text_lengths_histogram_afer_before.png")
        # Generar un boxplot que represente las longitudes de los datos de entrada
        boxplotTextLengths(df, "text_lengths_boxplot_before_cleaning.png")
        # Filtrar textos por longitud usando un umbral
        df = df[df['text_length']<=textLengthsFilter]
        # Volver a generar un histograma con el filtrado hecho
        histogramTextsLengths(df, "text_lengths_histogram_afer_cleaning.png")
        # Volver a generar un boxplot con el filtrado hecho
        boxplotTextLengths(df, "text_lengths_boxplot_after_cleaning.png")
        # Coger los textos y las etiquetas
    # Preprocesar datos
    if not preprocessedFile:
        print("[*] Preprocesando datos...")
        df = preprocesado(df, doc2vec_model, pca_model)
        embeddings = [point for point in df['text']]
        labels = np.asarray(df['class'])
    else:
        df = cargarDataset(preprocessedFile)
        # Eliminar \n para cargar los embeddings si los hay
        df['text'] = df['text'].replace('\n', '', regex=True)
        # Cargar los datos preprocesados en arrays
        embeddings = [eval(point) for point in df['text']]
        labels = np.asarray(df['class'])

    if not train:
        # TODO: arreglar predicciones
        df_original = cargarDataset(unpreprocessedFile)
        print("[*] Realizando prediciones...")
        with open(prediction_model, "rb") as file:
            model = pickle.load(file)
        for idx,text in enumerate(df['text']):
            print(f"Text: {df_original['text'][idx]}")
            print(f"preficted: {model.predict([text])[0]}")
            print()
        fileName = "predicted.csv"
        with open(f"{output_dir}/{fileName}", "w") as file:
            writer = csv.writer(file)
            writer.writerow(["text","class"])
            for idx,text in enumerate(df['text']):
                original_text = df_original['text'][idx]
                writer.writerow([original_text,model.predict([text])[0]])
        print(f"    Fichero guardado: {output_dir}/{fileName}")
        sys.exit(0)

    print("[*] Entrenando modelos...")        
    # Separar dataset en entrenamiento y pruebas
    X_train,X_test,y_train,y_test = train_test_split(embeddings,labels,test_size=0.2,random_state=42)

    # Guardar los scores de cada modelo de clasificacion en un array
    classification_scores = [] # (nombreClasificador, modelo, training_score, testing_score)

    # Clasificacion GaussianNB
    print(f"\t\t\tGaussianNB")
    print(f"{24*'-'}--------------{24*'-'}")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print(classification_report(y_test,y_pred))
    saveConfussionMatrix(cm, "d", "confussion_matrix_gaussianNB.png", cmap='summer')
    print()
    classification_scores.append(("GaussianNB", gnb, gnb.score(X_train,y_train), gnb.score(X_test,y_test)))

    """# Clasificacion MultinomialNB
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print(classification_report(y_test,y_pred))
    saveConfussionMatrix(cm, "d", "confussion_matrix_multinomialNB.png")"""
    
    # Clasificacion BernoulliNB
    print(f"\t\t\tBernoulliNB")
    print(f"{24*'-'}--------------{24*'-'}")
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    y_pred = bnb.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print(classification_report(y_test,y_pred))
    saveConfussionMatrix(cm, "d", "confussion_matrix_binomialNB.png", cmap='summer')
    print()
    classification_scores.append(("BernoulliNB", bnb, bnb.score(X_train,y_train), bnb.score(X_test,y_test)))

    # Clsificacion Random Forest
    print(f"\t\t\tRandomForest")
    print(f"{24*'-'}--------------{24*'-'}")
    rfc = RandomizedSearchCV(RandomForestClassifier(),{'n_estimators':[4,5],'criterion':['entropy'],
                                                      'max_depth':range(1,4),'min_samples_split':range(2,5)},random_state=12)
    rfc.fit(X_train, y_train)
    #print('Training score:',rfc.score(X_train, y_train))
    #print('Testing score:',rfc.score(X_test,y_test))
    print(rfc.best_estimator_)
    y_act=y_test
    y_pred=rfc.predict(X_test)
    cm = confusion_matrix(y_act,y_pred)
    print(classification_report(y_act,y_pred))
    saveConfussionMatrix(cm, "d", "confussion_matrix_RandomForest.png", cmap='Spectral')
    print()
    classification_scores.append(("RandomForest", rfc, rfc.score(X_train,y_train), rfc.score(X_test,y_test)))

    # Clsificacion Decision Tree
    print(f"\t\t\tDecisionTree")
    print(f"{24*'-'}--------------{24*'-'}")
    dtc= DecisionTreeClassifier(criterion='gini',splitter='random',min_samples_leaf=70,max_depth=4,random_state=0)
    dtc.fit(X_train, y_train)
    #print(model2.score(X_train, y_train))
    #print(model2.score(X_test,y_test))
    y_act=y_test
    y_pred=dtc.predict(X_test)
    cm = confusion_matrix(y_act,y_pred)
    print(classification_report(y_act,y_pred))
    saveConfussionMatrix(cm, "d", "confussion_matrix_DecisionTree.png", cmap='PiYG')
    print()
    classification_scores.append(("DecisionTree", dtc, dtc.score(X_train,y_train), dtc.score(X_test,y_test)))

    # TODO Meter mas modelos de prediccion

    # Clasificacion usando Ensemble Methods
    print(f"\t\t\tEnsembleMethods")
    print(f"{24*'-'}--------------{24*'-'}")
    ensemble_model = VotingClassifier(
        estimators=[("GaussianNB", GaussianNB()),
                    ("BernoulliNB", BernoulliNB()),
                    ("RandomForestClassifier", RandomForestClassifier()),
                    ("DecisionTreeClassifier", DecisionTreeClassifier())],
        voting='soft'  # 'hard' for majority voting, 'soft' for weighted voting based on class probabilities
    )
    # Fit the ensemble model on the training data
    ensemble_model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = ensemble_model.predict(X_test)
    cm = confusion_matrix(y_act,y_pred)
    print(classification_report(y_act,y_pred))
    saveConfussionMatrix(cm, "d", "confussion_matrix_EnsembleMethods.png", cmap='PiYG')
    print()
    classification_scores.append(("EnsembleMethods", ensemble_model,ensemble_model.score(X_train,y_train), ensemble_model.score(X_test,y_test)))

    # Guardar el modelo con mejor score
    print("[*] Guardando el mejor modelo")
    bestModel = None
    bestScore = 0
    for clasifier,modelo,train_score,test_score in classification_scores:
        score = (train_score+test_score)/2
        if score >= bestScore:
            bestModel = modelo
    fileName = f"{clasifier}_model.pkl"
    with open(fileName, "wb") as file:
        pickle.dump(bestModel, file)
    print(f"    Mejor modelo: {clasifier}")
    print(f"    Fichero guardado: {fileName}")


    """
    # Buscar el modelo con mejores hyperparametros:
    # Create a Random Forest Classifier
    rf_classifier = RandomForestClassifier()

    # Define the hyperparameter grid
    param_dist = {
        'n_estimators': randint(10, 200),
        'max_depth': randint(1, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'bootstrap': [True, False]
    }

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(rf_classifier, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)

    # Fit the model
    random_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print("Best Hyperparameters:", random_search.best_params_)

    # Get the best model
    best_model = random_search.best_estimator_

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy of the Best Model: {accuracy}")
    """
