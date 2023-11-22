# ClasificandoSuicidios
## Clonar el repositorio
```
git clone https://github.com/un4rch/ClasificandoSuicidios.git
```
## Activar entorno virtual
```
python3 -m venv clasificando_suicidios
source clasificando_suicidios/bin/activate
```
## Instalar dependencias
```
pip3 install -r requirements.txt
```
## Ejecutar la aplicacion
### Entrenar modelo (Datos SIN preprocesar)
```
python3 main.py config/trainUnprep_doc2vec.json # Usar doc2vec
python3 main.py config/trainUnprep_tfidf.json   # Usar tf-idf
```

### Entrenar modelo (Datos preprocesados)
```
python3 main.py config/trainPrep_doc2vec.json # Usar doc2vec
python3 main.py config/trainPrep_tfidf.json   # Usar tf-idf
```

### Realizar predicciones (Datos SIN preprocesar)
```
python3 main.py config/test_doc2vec.json # Usar doc2vec
python3 main.py config/test_tfidf.json   # Usar tf-idf
```
## Ejecucion personalizada
1. Crear un fichero llamado configFile.json (o usar el nombre que se quiera)
2. Copiar el siguiente contenido y configurar las variables
```
{
    "preprocessedFile": null, # str
    "unpreprocessedFile": null, # str
    "guardarPreproceso": null, # str
    "output_dir": null, # str
    "train": null, # true / false
    "visualization": null, # true / false
    "textLengthsFilter": null, # int
    "histogramIntervals": null, # int
    "preprocessType": null, # "tf-idf" / "doc2vec"
    "pca_dimensions": null, # int
    "doc2vec_vectors_size": null, # int
    "doc2vec_model": null, # str
    "pca_model": null, # str
    "tf_idf_model": null, # str
    "prediction_model": null, # str
    "max_num_samples": null # int
}
```
3. Ejecujar el siguiente comando:
```
python3 main.py config/configFile.json
```
## Desactivar entorno virtual
```
deactivate
```
