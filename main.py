# Visualizacion de embeddings: https://projector.tensorflow.org/
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra

# Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2,f_classif
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

import pickle
import string
"""
# System libraries
import os
import sys
# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
# Data processing libraries
import pandas as pd # frames (tables)
# NLP preprocessing libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from preprocessor import Preprocessor

#preprocessedFile = "Preprocessed_Suicide_Detection.csv"
preprocessedFile = None
unpreprocessedFile = "Suicide_Detection.csv"
savePreprocess = "Preprocessed_Suicide_Detection.csv"
output_dir = "output"
textLengthsFilter = 10000 # Se eliminan los textos cuya longitud sea mas de 10000 caracteres
histogramIntervals = 500

def cargarDataset(pFileName):
    # Cargar los datos
    data = pd.read_csv(pFileName)
    # Obtener 10000 instancias
    df = data.sample(n=10000, random_state=42)
    # Mostrar la informacion del dataframe y eliminar la columna que no nos interesa
    #print(df.info())
    df.drop(columns = 'Unnamed: 0',inplace=True)
    return df

def preprocesado(dataFrame):
    prerocessor = Preprocessor()
    for row in dataFrame['text'].values:
        print("----------------------------------------")
        print(row)
        linea = prerocessor.preprocesarLenguajeNatural(row, True)
        print(linea)
    print(linea)
    sys.exit(0)
    """# Convertir a minusculas
    dataFrame['text']= dataFrame['text'].str.lower()
    # Eliminar puntuaciones
    dataFrame['text'] = dataFrame['text'].str.replace(r'[^\w\s]+', '',regex = True)
    stop_words = stopwords.words('english')
    dataFrame['text'] = dataFrame['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    # Tokenizar
    dataFrame['text'] = dataFrame['text'].apply(lambda x:nltk.word_tokenize(x))
    # Lematizar palabras (stemming)
    ps = PorterStemmer()
    dataFrame['text'] = dataFrame['text'].apply(lambda x : [ps.stem(i) for i in x])
    dataFrame['text']=dataFrame['text'].apply(lambda x : ' '.join(x))"""
    # Eliminar si hay algun valor vacio
    #ind = dataFrame[dataFrame['text'].isnull()].index # indice de instancias vacias
    #print(dataFrame.iloc[ind]) # imprimir contenido de las instancias vacias
    dataFrame.dropna(inplace=True)
    # PCA space dim reduction
    # TODO
    # Document embeddings
    # TODO
    # Guardar el preproceso
    dataFrame.to_csv("Preprocessed_Suicide_Detection.csv")
    return dataFrame

def histogramTextsLengths(dataFrame, fileName):
    dataFrame['text_lengths'] = dataFrame['text'].apply(len)
    # Create a histogram with intervals of 500
    plt.hist(dataFrame['text_lengths'], bins=range(0, max(dataFrame['text_lengths']) + histogramIntervals + 1, histogramIntervals), edgecolor='black')

    # Set labels and title
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Text Lengths')

    # Save the plot as a PNG file
    plt.savefig(f"{output_dir}/{fileName}")
    plt.close()

def boxplotTextLengths(dataFrame, fileName):
    # Function to calculate length of each item
    def calculate_length(item):
        return len(str(item))
    dataFrame['text_lengths'] = dataFrame['text'].apply(calculate_length)
    #print(dataFrame['text_lengths'].sum())
    sns.boxplot(x="text_lengths", y="class", data=dataFrame)
    plt.savefig(f"{output_dir}/class_separated_{fileName}")
    plt.close()
    sns.boxplot(x="text_lengths", data=dataFrame)
    plt.savefig(f"{output_dir}/{fileName}")
    plt.close()

    # resetear indices
    #textLenghts = textLenghts.reset_index(drop=True)
    """
    textLenghts = []
    for row in df.values:
        # Access row data using row[index]
        print(row[0])
        print(row[1])
        print("-----------")
    for idx,row in dataFrame.iterrows():
        text = row['text']
        label = row['class']
        textLenghts.append(len(text))
    #sns.boxplot()
    print(sum(textLenghts))
    """

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
    
if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not preprocessedFile:
        df = cargarDataset(unpreprocessedFile)
        df = preprocesado(df)
    else:
        df = cargarDataset(preprocessedFile)
    # Eliminar si ha quedado algun valor vacio al cargar el dataset
    df.dropna(inplace=True)
    # Visualizar si los datos estan balanceados
    visualizeBalanced(df, "check_balanced.png")
    # Generar un histograma que muestra en numero de instancias que tienen cierta cantidad de letras por intervalos
    histogramTextsLengths(df, "text_lengths_histogram_afer_before.png")
    # Generar un boxplot que represente las longitudes de los datos de entrada
    boxplotTextLengths(df, "text_lenghts_boxplot_before_cleaning.png")
    # Filtrar textos por longitud usando un umbral
    df = df[df['text_lengths']<=textLengthsFilter]
    # Volver a generar un histograma con el filtrado hecho
    histogramTextsLengths(df, "text_lengths_histogram_afer_cleaning.png")
    # Volver a generar un boxplot con el filtrado hecho
    boxplotTextLengths(df, "text_lenghts_boxplot_after_cleaning.png")
    # Coger los textos y las etiquetas
    x,y = df['text'],df['class']
