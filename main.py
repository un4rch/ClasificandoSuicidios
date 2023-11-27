# Algoritmos: 
#import ast
# Warnings
import warnings
warnings.filterwarnings('ignore')
# System libraries
import os
import sys
import csv
# Scripting
#import typer
#from rich.table import Table
#from rich.console import Console
from tabulate import tabulate
# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
# Data processing libraries
import pandas as pd # frames (tables)
import numpy as np # linear algebra
#import string
import json
# NLP preprocessing libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from preprocessor import Preprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
import nlpaug.augmenter.char as textAugmenter
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
from sklearn.metrics import f1_score

preprocessedFile = None
unpreprocessedFile = None
guardarPreproceso = None
output_dir = None
train = None
visualization = None
textLengthsFilter = None
histogramIntervals = None
preprocessType = None # doc2vec,tf-idf
pca_dimensions = None
doc2vec_vectors_size = None
doc2vec_model = None
pca_model = None
tf_idf_model = None
prediction_model = None
max_num_samples = None
usarRuido = None

def printPredictionsTable(predictionsDict):
    # expected input: [["text1", "prediction"], ["text2", "prediction"], ...]
    print (tabulate(predictionsDict, headers=["Original Text", "Prediction"], tablefmt="simple_outline"))

def inicializarPrograma(configFile):
    with open(configFile, "r") as file:
        data = json.load(file)
        file.close()
    global preprocessedFile
    preprocessedFile = data['preprocessedFile']
    global unpreprocessedFile
    unpreprocessedFile = data['unpreprocessedFile']
    global guardarPreproceso
    guardarPreproceso = data['guardarPreproceso']
    global output_dir
    output_dir = data['output_dir']
    global train
    train = data['train']
    global visualization
    visualization = data['visualization']
    global textLengthsFilter
    textLengthsFilter = data['textLengthsFilter']
    global histogramIntervals
    histogramIntervals = data['histogramIntervals']
    global preprocessType
    preprocessType = data['preprocessType']
    global pca_dimensions
    pca_dimensions = data['pca_dimensions']
    global doc2vec_vectors_size
    doc2vec_vectors_size = data['doc2vec_vectors_size']
    global doc2vec_model
    doc2vec_model = data['doc2vec_model']
    global pca_model
    pca_model = data['pca_model']
    global tf_idf_model
    tf_idf_model = data['tf_idf_model']
    global prediction_model
    prediction_model = data['prediction_model']
    global max_num_samples
    max_num_samples = data['max_num_samples']
    global usarRuido
    usarRuido = data['usarRuido']
    

def cargarDataset(pFileName):
    # Cargar los datos
    df= pd.read_csv(pFileName)
    # Coger solo las columnas que nos interesan
    #df = pd.DataFrame()
    #df['text'] = df_tmp['text']
    #df['class'] = df_tmp['class']
    return df

def preprocesarNLP(pdColumn):
    pdColumn= pdColumn.str.lower()
    pdColumn = pdColumn.str.replace(r'[^\w\s\d+]+', '',regex = True)
    stop_words = stopwords.words('english')
    pdColumn = pdColumn.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    pdColumn = pdColumn.apply(lambda x:nltk.word_tokenize(x))
    ps = PorterStemmer()
    pdColumn = pdColumn.apply(lambda x : [ps.stem(i) for i in x])
    pdColumn=pdColumn.apply(lambda x : ' '.join(x))
    pdColumn.dropna(inplace=True)
    return pdColumn
def preprocesado(dataFrame, doc2vec_model, pca_model, tfIdf_model):
    # Filtrar textos por longitud usando un umbral
    if textLengthsFilter:
        dataFrame['text_length'] = dataFrame['text'].apply(len)
        dataFrame = dataFrame[dataFrame['text_length']<=textLengthsFilter]
    # Preprocesamos los datos
    x_prep = None
    y_prep = None
    preprocessor = Preprocessor()
    if preprocessType == "doc2vec":
        x_prep,y_prep,doc2vec_model,pca_model = preprocessor.doc2vec(dataFrame['text'], dataFrame['class'], pca_dimensions=pca_dimensions, doc2vec_vectors_size=doc2vec_vectors_size, doc2vec_model=doc2vec_model, pca_model=pca_model)
        # Convertir de np.array a tuplas
        x_prep = [tuple(point) for point in x_prep.tolist()]
        # Guardar el preproceso en el dataFrame
        dataFrame['text'] = x_prep
        dataFrame['class'] = y_prep
        if train and guardarPreproceso:
            # Guardar los modelos de d0c2vec y pca
            doc2vec_model.save(unpreprocessedFile.split(".")[0]+"_doc2vec.model")
            with open(unpreprocessedFile.split(".")[0]+"_pca.model", "wb") as file:
                pickle.dump(pca_model, file)
            print(f"    Fichero guardado: {unpreprocessedFile.split('.')[0]+'_doc2vec.model'}")
            print(f"    Fichero guardado: {unpreprocessedFile.split('.')[0]+'_pca.model'}")
            if guardarPreproceso != None:
                dataFrame.to_csv(guardarPreproceso, index=False)
                print(f"    Fichero guardado: {guardarPreproceso}")
    elif preprocessType == "tf-idf":
        # Preprocesar lenguaje natural
        dataFrame['text'] = preprocesarNLP(dataFrame['text'])
        if not tfIdf_model:
            vectorizer = TfidfVectorizer(min_df=50,max_features=5000)
            tfidf_matrix =  vectorizer.fit_transform(dataFrame['text']).toarray()
            x_prep = pd.DataFrame(tfidf_matrix, columns=vectorizer.get_feature_names_out())
            x_prep = x_prep.drop("class", axis=1)
            y_prep = np.asarray(dataFrame['class'])
            # Si no se hace copy se trata como un puntero y luego x_prep tendra la columna "class"
            dataFrame = x_prep.copy()
            dataFrame['class'] = y_prep
        else:
            with open(tfIdf_model, "rb") as file:
                vectorizer_vocabulary = pickle.load(file)
            vectorizer = TfidfVectorizer(vocabulary=vectorizer_vocabulary)
            x_prep =  vectorizer.fit_transform(dataFrame['text']).toarray()
            x_prep = pd.DataFrame(x_prep, columns=vectorizer.get_feature_names_out())
            x_prep = x_prep.drop("class", axis=1)
            y_prep = np.asarray(dataFrame['class'])
            #dataFrame = x_prep
            #dataFrame['class'] = y_prep
        if train and guardarPreproceso:
            with open(unpreprocessedFile.split(".")[0]+"_tf_idf.model", 'wb') as f:
                pickle.dump(vectorizer.vocabulary_, f) 
            print(f"    Fichero guardado: {unpreprocessedFile.split('.')[0]+'_tf_idf.model'}")
            dataFrame.to_csv(guardarPreproceso, index=False)
            print(f"    Fichero guardado: {guardarPreproceso}")
    # Guardar el preproceso de los datos
    return x_prep, y_prep

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
    classCnt = dataFrame['class'].value_counts()
    #print(classCnt)

    plt.figure(figsize=((20,5)))

    plt.subplot(1,2,1)
    sns.countplot(dataFrame,x='class')

    plt.subplot(1,2,2)
    plt.pie(classCnt,labels = classCnt.index,autopct='%.0f%%')

    plt.savefig(f"{output_dir}/{fileName}")
    plt.close()
    print(f"    Fichero guardado: {output_dir}/{fileName}")

def saveConfussionMatrix(confussion_matrix, dType, fileName, cmap):
    plt.figure()
    sns.heatmap(confussion_matrix, annot=True, fmt=dType, cmap=cmap)
    plt.title(fileName.split(".")[0])
    plt.ylabel("Real")
    plt.xlabel("Predicted")
    plt.savefig(f"{output_dir}/{fileName}", format='png')
    plt.close()
    print(f"    Fichero guardado: {output_dir}/{fileName}")

def visualizarDatosEntrada(dataFrame):
        def visualizarGraficas(dataFrame, filtro):
            # Visualizar si los datos estan balanceados despues de filtrar
            visualizeBalanced(dataFrame, f"check_balanced_{filtro}_filter.png")
            # Volver a generar un histograma con el filtrado hecho
            histogramTextsLengths(dataFrame, f"text_lengths_histogram_{filtro}_cleaning.png")
            # Generar un boxplot en funcion de las longitudes de los textos
            boxplotTextLengths(dataFrame, f"text_lengths_boxplot_{filtro}_cleaning.png")
        # Mostrar error si los datos de df son embeddings y no los textos originales
        if preprocessedFile:
            print("[!] Error para visualizar los datos de entrada, los datos no deben estar preprocesados")
            sys.exit(1)
        # Calcular una columna de longitudes de textos    
        dataFrame['text_length'] = dataFrame['text'].apply(len)
        # Generar graficas
        print("[*] Generando graficas de visualizacion de los datos...")
        # Sin hacer ningun filtro
        visualizarGraficas(dataFrame, 0)
        # Filtrar textos por longitud usando un umbral
        dataFrame = dataFrame[dataFrame['text_length']<=textLengthsFilter]
        visualizarGraficas(dataFrame, textLengthsFilter)

def barClassifiersFScores(classifierNames, classifierScores, fileName):
    plt.figure(figsize =(18, 7))
    plt.bar(classifierNames, classifierScores, color="Blue", width=0.8)
    for modelName,modelScore in zip(classifierNames, classifierScores):
        plt.text(modelName, modelScore, ("%.4f" % modelScore), fontsize=10, ha='center', va='bottom')
    plt.xlabel("Classifier")
    plt.ylabel("F-Score")
    plt.title("Classifiers f-scores")
    plt.savefig(f"{output_dir}/{fileName}")
    plt.close()
    print(f"    Fichero guardado: {output_dir}/{fileName}")

def evaluarRuido(X_ruido, y_ruido):
    print("[*] Evaluando ruido...")
    # Evaluar ruido
    fileName = "ruido_fscores.png"
    with open(prediction_model, "rb") as file:
        model = pickle.load(file)
    # for con diferentes porcentajes de ruido
    fscores = []
    actions = []
    for actionType in ["substitute", "insert"]:
        for ruido in [0.05,0.1,0.2,0.25,0.5]:
            aug = textAugmenter.RandomCharAug(action=actionType, aug_char_p=ruido)
            X_act = []
            for texto in X_ruido:
                X_act.append(aug.augment(texto)[0])
            dataFrame = pd.DataFrame()
            dataFrame['text'] = X_act
            dataFrame['class'] = np.asarray(y_ruido)
            x_prep, y_prep = preprocesado(dataFrame, doc2vec_model, pca_model, tf_idf_model)
            X_act = x_prep
            y_pred = model.predict(X_act)
            fScore = f1_score(y_prep, y_pred, average='weighted')
            actions.append(f"{actionType}_{ruido}")
            fscores.append(fScore)
    plt.figure(figsize =(15, 7))
    plt.bar(actions, fscores, color="Blue", width=0.8)
    for tipoRuido,fScore in zip(actions, fscores):
        plt.text(tipoRuido, fScore, ("%.4f" % fScore), fontsize=10, ha='center', va='bottom')
    plt.xlabel("Ruido")
    plt.ylabel("F-Score")
    plt.title("Evolucion f-score con diferentes ruidos")
    plt.savefig(f"{output_dir}/{fileName}")
    plt.close()
    print(f"    Fichero guardado: {output_dir}/{fileName}")

def evaluarRuidoGraficoLineas(dataFrame):
    X_train,X_test,y_train,y_test = train_test_split(dataFrame['text'],dataFrame['class'],test_size=0.2,random_state=42)
    modelos = [("GaussianNB", GaussianNB()),
               ("MultinomialNB", MultinomialNB()),
               ("BernoulliNB", BernoulliNB()),
               ("RandomForestClassifier", RandomizedSearchCV(RandomForestClassifier(),{'n_estimators':[4,5],'criterion':['entropy'],
                                                      'max_depth':range(1,4),'min_samples_split':range(2,5)},random_state=12)),
               ("DecisionTreeClassifier", DecisionTreeClassifier()),
               ("EnsembleMethods", VotingClassifier(
                    estimators=[("GaussianNB", GaussianNB()),
                        ("BernoulliNB", BernoulliNB()),
                        ("RandomForestClassifier", RandomForestClassifier()),
                        ("DecisionTreeClassifier", DecisionTreeClassifier())],
                    voting='soft'  # 'hard' for majority voting, 'soft' for weighted voting based on class probabilities
                )),
                ("EnsembleMethodsGaussian", VotingClassifier(
                    estimators=[("GaussianNB", GaussianNB()),
                        ("MultinomialNB", MultinomialNB()),
                        ("BernoulliNB", BernoulliNB()),],
                    voting='soft'  # 'hard' for majority voting, 'soft' for weighted voting based on class probabilities
                ))
    ]
    plt.figure(figsize=(15,7))
    actions = []
    for nombreModelo,modelo in modelos:
        fscores = []
        for actionType in ["insert"]:
            for ruido in [0.05,0.1,0.2,0.25,0.5,0.75]:
                if f"{actionType}_{ruido}" not in actions:
                    actions.append(f"{actionType}_{ruido}")
                print(f"Evaluando: {actionType}_{ruido}_{nombreModelo}")
                aug = textAugmenter.RandomCharAug(action=actionType, aug_char_p=ruido)
                X_train_tmp = []
                for texto in X_train:
                    X_train_tmp.append(aug.augment(texto)[0])
                dataFrame_train = pd.DataFrame()
                dataFrame_train['text'] = X_train_tmp
                dataFrame_train['class'] = np.asarray(y_train)
                X_train_ruido, y_train_ruido = preprocesado(dataFrame_train, doc2vec_model, pca_model, tf_idf_model)
                X_test_tmp = []
                for texto in X_test:
                    X_test_tmp.append(aug.augment(texto)[0])
                dataFrame_test = pd.DataFrame()
                dataFrame_test['text'] = X_test_tmp
                dataFrame_test['class'] = np.asarray(y_test)
                X_test_ruido, y_test_ruido = preprocesado(dataFrame_test, doc2vec_model, pca_model, tf_idf_model)
                modelo.fit(X_train_ruido, y_train_ruido)
                y_pred = modelo.predict(X_test_ruido)
                fscores.append(f1_score(y_test_ruido, y_pred, average='weighted'))
        plt.plot(actions, fscores, label=nombreModelo)
    # Agregar etiquetas y título
    plt.xlabel('Ruido')
    plt.ylabel('F-score')
    plt.title('Ruido con diferentes modelos de prediccion')
    # Agregar leyenda
    plt.legend()
    # Mostrar el gráfico
    fileName = "ruidoModelos.png"
    plt.savefig(f"{output_dir}/{fileName}")
    plt.close()
    print(f"Imagen guardada: {output_dir}/{fileName}")
    


if __name__ == "__main__":
    # Inicializar las variables del programa
    inicializarPrograma(sys.argv[1])
    # Crear directorio donde se van a guardar las graficas y ficheros generados
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("[*] Cargarndo datos...")
    
    # Preprocesar datos
    if not preprocessedFile:
        # Cargar dataset sin preprocesar
        df = cargarDataset(unpreprocessedFile)
        # Eliminar si ha quedado algun valor vacio al cargar el dataset
        df.dropna(inplace=True)
        # Coger un subconjunto de datos
        if max_num_samples:
            if df.shape[0] > max_num_samples:
                df = df.sample(n=max_num_samples, random_state=42)
        # Visualizar datos de entrada
        if visualization:
            if usarRuido:
                # Evaluar ruido para diferentes modelos de prediccion
                print("[*] Evaluando ruido para diferentes modelos de clasificacion...")
                gptmp = guardarPreproceso
                guardarPreproceso = False
                evaluarRuidoGraficoLineas(df)
                guardarPreproceso = gptmp
            # Separar dataset en entrenamiento y pruebas
            X_train,X_test,y_train,y_test = train_test_split(df['text'],df['class'],test_size=0.2,random_state=42)
            df_train = pd.DataFrame()
            df_train['text'] = X_train
            df_train['class'] = y_train
            df_test = pd.DataFrame()
            df_test['text'] = X_test
            df_test['class'] = y_test
            df_tn = df_train.copy()
            df_tt = df_test.copy()
            visualizarDatosEntrada(df)
            df_tn['text_length'] = df_tn['text'].apply(len)
            df_tn = df_tn[df_tn['text_length']<=2000]
            df_tt['text_length'] = df_tt['text'].apply(len)
            df_tt = df_tt[df_tt['text_length']<=2000]
            boxplotTextLengths(df_tt, f"text_lengths_boxplot_{2000}_train.png")
            boxplotTextLengths(df_tn, f"text_lengths_boxplot_{2000}_test.png")
            if usarRuido:
                df = df.sample(n=max_num_samples, random_state=42)
                X_temp,X_ruido,y_temp,y_ruido = train_test_split(df["text"],df['class'],test_size=0.5,random_state=42)
                guardarPreproceso = False
                evaluarRuido(X_ruido, y_ruido)
                guardarPreproceso = True
                df = pd.DataFrame()
                df['text'] = X_temp
                df['class'] = y_temp
        # Calcular una columna que contenga las longitudes de los textos (sirve para el preprocesado)
        print("[*] Preprocesando datos...")
        x_prep, y_prep = preprocesado(df, doc2vec_model, pca_model, tf_idf_model)
        X_train,X_test,y_train,y_test = train_test_split(x_prep,y_prep,test_size=0.2,random_state=42)
        #x_prep, y_prep = preprocesado(df, doc2vec_model, pca_model, tf_idf_model)
        #data = x_prep
        #labels = y_prep
    else:
        df = cargarDataset(preprocessedFile)
        # Eliminar si ha quedado algun valor vacio al cargar el dataset
        df.dropna(inplace=True)
        data = df['text']
        labels = df["class"]
        # Separar dataset en entrenamiento y pruebas
        X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.2,random_state=42)
    
    # Adaptar el formato de los datos para cada tipo de preprocesado
    if preprocessType == "doc2vec":
        try:
            X_train = [eval(embedding) for embedding in X_train]
            X_test = [eval(embedding) for embedding in X_test]
        except:
            X_train = [tuple(embedding) for embedding in X_train]
            X_test = [tuple(embedding) for embedding in X_test]
    elif preprocessType == "tf-idf":
        X_train = [row for idx,row in X_train.iterrows()]
        X_test = [row for idx,row in X_test.iterrows()]

    if not train:
        df_original = cargarDataset(unpreprocessedFile)
        print("[*] Realizando prediciones...")
        print()
        with open(prediction_model, "rb") as file:
            model = pickle.load(file)
        predictionsTable = []
        for idx,embedded_text in enumerate(data):
            original_text = df_original['text'][idx]
            prediction = model.predict([embedded_text])[0]
            predictionsTable.append([original_text, prediction])
        printPredictionsTable(predictionsTable)
        fileName = "predicted.csv"
        with open(f"{output_dir}/{fileName}", "w") as file:
            writer = csv.writer(file)
            writer.writerow(["text","class"])
            for idx,text in enumerate(data):
                original_text = df_original['text'][idx]
                writer.writerow([original_text,model.predict([text])[0]])
        print()
        print(f"    Fichero guardado: {output_dir}/{fileName}")
        sys.exit(0)

    print("[*] Entrenando modelos...")
    
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
    #classification_scores.append(("GaussianNB", gnb, gnb.score(X_train,y_train), gnb.score(X_test,y_test)))
    classification_scores.append(("GaussianNB", gnb, f1_score(y_test, y_pred, average='weighted')))

    # Clasificacion MultinomialNB
    print(f"\t\t\tMultinomialNB")
    print(f"{24*'-'}--------------{24*'-'}")
    mnb = MultinomialNB()
    if preprocessType == "doc2vec":
        correccion = 10 # El valor mas negativo suele oscilar alrededor del 4, 5 nos asegura que es mayor
        X_train = [[x + correccion for x in vector] for vector in X_train]
        X_test = [[x + correccion for x in vector] for vector in X_test]
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print(classification_report(y_test,y_pred))
    saveConfussionMatrix(cm, "d", "confussion_matrix_multinomialNB.png", cmap='summer')
    print()
    #classification_scores.append(("MultinomialNB", mnb, mnb.score(X_train,y_train), mnb.score(X_test,y_test)))
    classification_scores.append(("MultinomialNB", mnb, f1_score(y_test, y_pred, average='weighted')))
    
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
    #classification_scores.append(("BernoulliNB", bnb, bnb.score(X_train,y_train), bnb.score(X_test,y_test)))
    classification_scores.append(("BernoulliNB", bnb, f1_score(y_test, y_pred, average='weighted')))

    # Clsificacion Random Forest
    print(f"\t\t\tRandomForest")
    print(f"{24*'-'}--------------{24*'-'}")
    rfc = RandomizedSearchCV(RandomForestClassifier(),{'n_estimators':[4,5],'criterion':['entropy'],
                                                      'max_depth':range(1,4),'min_samples_split':range(2,5)},random_state=12)
    rfc.fit(X_train, y_train)
    #print('Training score:',rfc.score(X_train, y_train))
    #print('Testing score:',rfc.score(X_test,y_test))
    print(rfc.best_estimator_)
    y_pred=rfc.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print(classification_report(y_test,y_pred))
    saveConfussionMatrix(cm, "d", "confussion_matrix_RandomForest.png", cmap='Spectral')
    print()
    #classification_scores.append(("RandomForest", rfc, rfc.score(X_train,y_train), rfc.score(X_test,y_test)))
    classification_scores.append(("RandomForest", rfc, f1_score(y_test, y_pred, average='weighted')))

    # Clsificacion Decision Tree
    print(f"\t\t\tDecisionTree")
    print(f"{24*'-'}--------------{24*'-'}")
    dtc= DecisionTreeClassifier(criterion='gini',splitter='random',min_samples_leaf=70,max_depth=4,random_state=0)
    dtc.fit(X_train, y_train)
    #print(model2.score(X_train, y_train))
    #print(model2.score(X_test,y_test))
    y_pred=dtc.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print(classification_report(y_test,y_pred))
    saveConfussionMatrix(cm, "d", "confussion_matrix_DecisionTree.png", cmap='PiYG')
    print()
    #classification_scores.append(("DecisionTree", dtc, dtc.score(X_train,y_train), dtc.score(X_test,y_test)))
    classification_scores.append(("DecisionTree", dtc, f1_score(y_test, y_pred, average='weighted')))

    # TODO Meter mas modelos de prediccion

    # Clasificacion usando Ensemble Methods general
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
    cm = confusion_matrix(y_test,y_pred)
    print(classification_report(y_test,y_pred))
    saveConfussionMatrix(cm, "d", "confussion_matrix_EnsembleMethods.png", cmap='PiYG')
    print()
    #classification_scores.append(("EnsembleMethods", ensemble_model, ensemble_model.score(X_train,y_train), ensemble_model.score(X_test,y_test)))
    classification_scores.append(("EnsembleMethods", ensemble_model, f1_score(y_test, y_pred, average='weighted')))

    # Clasificacion usando Ensemble Methods bayesianos
    print(f"\t\t\tEnsembleMethodsGaussian")
    print(f"{24*'-'}--------------{24*'-'}")
    ensemble_model = VotingClassifier(
        estimators=[("GaussianNB", GaussianNB()),
                    ("MultinomialNB", MultinomialNB()),
                    ("BernoulliNB", BernoulliNB()),],
        voting='soft'  # 'hard' for majority voting, 'soft' for weighted voting based on class probabilities
    )
    # Fit the ensemble model on the training data
    ensemble_model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = ensemble_model.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print(classification_report(y_test,y_pred))
    saveConfussionMatrix(cm, "d", "confussion_matrix_EnsembleMethodsBayesian.png", cmap='PiYG')
    print()
    #classification_scores.append(("EnsembleMethods", ensemble_model, ensemble_model.score(X_train,y_train), ensemble_model.score(X_test,y_test)))
    classification_scores.append(("EnsembleMethodsBayesian", ensemble_model, f1_score(y_test, y_pred, average='weighted')))

    # Generar un grafico de barras con los fscores de cada clasificador
    classifiers = [classifier_score[0] for classifier_score in classification_scores]
    scores = [classifier_score[2] for classifier_score in classification_scores]
    print("[*] Generando metricas de los diferentes modelos")
    barClassifiersFScores(classifiers, scores, "classifiers_fscores.png")

    # Guardar el modelo con mejor fscore
    print("[*] Guardando el mejor modelo...")
    classifier,bestModel,fscore = max(classification_scores, key =  lambda i : i[2])
    fileName = f"{classifier}_model_{preprocessType}.pkl"
    with open(fileName, "wb") as file:
        pickle.dump(bestModel, file)
    print(f"    Mejor modelo: {classifier}")
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
