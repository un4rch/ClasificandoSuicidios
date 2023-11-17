import numpy as np
import pandas as pd
import gensim
import nltk
import re
import emoji
import emot
import sys
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pickle

class Preprocessor:
    def __init__(self) -> None:
          pass
    
    def convertirEmojis(self, texto, switch):  # Convierte un emoji en un conjunto de palabras en inglés que lo representa. Si switch es False, entonces se eliminan los emojis
        if switch:
            texto = emoji.demojize(texto)
            diccionario_emojis = emot.emo_unicode.EMOTICONS_EMO
            for emoticono, texto_emoji in diccionario_emojis.items():
                texto = texto.replace(emoticono, texto_emoji)
        else:
            texto = emoji.demojize(texto)
            texto = re.sub(":.*?:", "", texto)
            texto = texto.strip()
        return texto
    
    def eliminarSignosPuntuacion(self, texto):  # Dado un string, devuelve el mismo string eliminando todos los caracteres que no sean alfabéticos
        textoNuevo = ""
        for caracter in texto:  # Por cada caracter en el texto
            if caracter == "_":  # Si es una barra baja, entonces se traduce como espacio
                textoNuevo = textoNuevo + " "
            if caracter.isalpha() or caracter == " ":  # Si pertenece al conjunto de letras del alfabeto, se engancha a "textoNuevo"
                textoNuevo = textoNuevo + caracter
        return(textoNuevo)
    
    def eliminarStopWords(self, texto):  # Dado un string, elimina las stopwords de ese string
        texto = word_tokenize(texto, language='english')
        textoNuevo = ""
        for palabra in texto:
            if palabra not in stopwords.words('english'):
                textoNuevo = textoNuevo + " " + palabra
        return(textoNuevo)

    def normalizarTexto(self, texto):  # Dado un string que contenga palabras, devuelve un string donde todas las letras sean minúsculas
        return(texto.lower())
    
    def aux_lematizar(self, palabra):
        tag = nltk.pos_tag([palabra])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def lematizar(self, texto):  # Dado un string, lematiza las palabras de ese string
        texto = nltk.word_tokenize(texto)

        # Inicializar el lematizador
        lemmatizer = WordNetLemmatizer()

        # Lematizar cada palabra y agregarla a una lista
        palabras_lematizadas = []
        for palabra in texto:
            pos = self.aux_lematizar(palabra)
            palabra_l = lemmatizer.lemmatize(palabra, pos=pos)
            palabras_lematizadas.append(palabra_l)

        # Unir las palabras lematizadas en un solo string y devolverlo
        texto_lematizado = ' '.join(palabras_lematizadas)
        return texto_lematizado
    
    def preprocesarLenguajeNatural(self, pColumna, pSwitch):  # Realiza todo el preproceso de un string en el orden correcto
        linea = str(pColumna)
        linea = self.convertirEmojis(linea, pSwitch)
        linea = self.eliminarSignosPuntuacion(linea)
        linea = self.normalizarTexto(linea)
        linea = self.eliminarStopWords(linea)
        linea = self.lematizar(linea)
        return linea
    
    def doc2vec(self, texts_array, labels_array, pca_dimensions, doc2vec_vectors_size=None, doc2vec_model=None, pca_model=None):
        texts_array = np.array(texts_array)
        y = np.array(labels_array)

        # Limpia y tokeniza el texto de entrada
        x_cleaned = [self.preprocesarLenguajeNatural(t, "switch") for t in texts_array]
        x_tokenized = [[w for w in sentence.split(" ") if w != ""] for sentence in x_cleaned]

        # Mapea las etiquetas únicas a valores numéricos
        if not pd.isnull(labels_array).all():
            label_map = {cat:index for index,cat in enumerate(np.unique(labels_array))}
            y_prep = np.asarray([label_map[l] for l in labels_array])
        else:
            y_prep = None

        # Entrena un modelo Doc2Vec si no se proporciona uno    
        if not doc2vec_model:
            # Crea datos etiquetados para el modelo Doc2Vec
            tagged_data = [TaggedDocument(words=row, tags=[str(label)]) for row, label in zip(x_tokenized, y_prep)]
            # Configura y entrena el modelo Doc2Vec
            model = Doc2Vec(vector_size=doc2vec_vectors_size, window=5, min_count=1, workers=4, epochs=20)
            model.build_vocab(tagged_data)
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        else:
            # Carga un modelo Doc2Vec existente si se proporciona uno
            model = Doc2Vec.load(doc2vec_model)

        # Genera vectores de coordenadas para los datos de entrada
        nuevo_vectors = []
        for tokens in x_tokenized:
            vector = model.infer_vector(tokens)
            nuevo_vectors.append(vector)

        # Realiza reducción de dimensionalidad (PCA) si no se proporciona un modelo PCA    
        if not pca_model:
            pca_model = PCA(n_components=pca_dimensions)
            pca_model.fit(nuevo_vectors)
        else:
            with open(pca_model, 'rb') as file:
                pca_model = pickle.load(file)
        x_prep = pca_model.transform(nuevo_vectors)

        # Devuelve los datos procesados, etiquetas, modelo Doc2Vec y modelo PCA
        return x_prep, y_prep, model, pca_model
