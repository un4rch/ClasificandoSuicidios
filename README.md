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
python3 main.py config/trainUnprepConfig.json
```

### Entrenar modelo (Datos preprocesados)
```
python3 main.py config/trainPrepConfig.json
```

### Realizar predicciones (Datos SIN preprocesar)
```
python3 main.py config/testConfig.json
```
## Desactivar entorno virtual
```
deactivate
```
