# IdentificacionRetinopatia
## Identificación de retinopatías

El Propósito del siguiente trabajo es identificar los pacientes que tienen complicaciones diabéticas, como lo son la neuropatía, nefropatía y retinopatía de notas médicas. Es el trabajo final del curso Clinical Natural Language Processing impartido en Coursera. Las notas medicas se encuentran en el siguiente linklink para su entrenamiento del modelo:

  https://raw.githubusercontent.com/hhsieh2416/Identify_Diabetic_Complications/main/data/diabetes_notes.csv
  
Y los datos para su validación se encuentran en el siguiente link:
  https://raw.githubusercontent.com/hhsieh2416/Identify_Diabetic_Complications/main/data/glodstandrad.csv
  
En primera instancia, se crea el siguiente código para ignorar los warnings:
```python 
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')
datos = "https://raw.githubusercontent.com/hhsieh2416/Identify_Diabetic_Complications/main/data/diabetes_notes.csv"
df = pd.read_csv(datos)
# Importando las paqueterías necesarias:
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
# Lectura de datos
datos = "https://raw.githubusercontent.com/hhsieh2416/Identify_Diabetic_Complications/main/data/diabetes_notes.csv"
df = pd.read_csv(datos)
# Análisis grafico de los datos
fig, ax = plt.subplots()
ax.bar(df['NOTE_ID'],df['TEXT'].str.split().apply(len))

# Cantidad de palabras por reporte de cada paciente identificado por un id
conteo = df['TEXT'].str.split().apply(len).tolist()
print('Media de palabras: ' + str(np.mean(conteo)))
print('Mediana de palabras: ' + str(np.median(conteo)))
print('Minimo de palabras: ' + str(np.min(conteo)))
print('Maximo de palabras: ' + str(np.max(conteo)))

def reporte_paciente(id):
    resumen = re.findall(r"\w+", str(df[df.NOTE_ID == id]['TEXT'].tolist() ))
    return resumen
# print(reporte_paciente(1))

```
Ahora bien, se genera una función la cual recibe nuestro DataFrame con las notas médicas, la palabra a buscar y el tamaño de la ventana
## Función sin expresiones regulares
```python 

def extract_text_window(df, word, window_size, column_name = "TEXT"):
    
    #Constants
    user_input = f'({word})'
    regex = re.compile(user_input)
    
    negative = f'(no history of {word}|No history of {word}|any comorbid complications|family history|father also has {word}|denies {word}|Negative for {word})'
    regex_negative = re.compile(negative)
    
    half_window_size = window_size 
    final_df = pd.DataFrame([])
    column_position = df.columns.get_loc(column_name) + 1  #We add 1 cause position 0 is the index
    
    
    #Loop for each row of the column
    for row in df.itertuples():
        
        #Loop for multiple matches in the same row
        for match in regex.finditer(row[column_position]):
            window_start = int([match.start()-half_window_size if match.start()>=half_window_size else 0][0])
            window_end = int([match.end() + half_window_size if match.end()+half_window_size <= len(row[column_position]) else len(row[column_position])][0])
            
            final_df = final_df.append({
                                        "WORD": match.group(),
                                        "START_INDEX": match.start(),
                                        "WINDOW_START": window_start,
                                        "WINDOW_END": window_end,
                                        "CONTEXT": row[column_position][window_start:window_end],
                                        "FULL_TEXT": row[column_position],
                                        "NOTE_ID": row[1]},
                                        ignore_index=True)
    #Extracción de negativos
        for match in regex_negative.finditer(row[column_position]):
            final_df2 = final_df[final_df["CONTEXT"].str.contains(pat = regex_negative, regex = True)==False]
    return "No matches for the pattern" if len(final_df) == 0 else  final_df2
    
    
# Buscando diabet en las notas médicas
df = pd.read_csv("https://raw.githubusercontent.com/hhsieh2416/Identify_Diabetic_Complications/main/data/diabetes_notes.csv")
word = "diabet"
window_size = 50 #tamaño de la ventana
diabetes_notes_window = extract_text_window(df,word,window_size)

diabetes_notes_window 
```
Se crea una segunda función la cual recibe nuestro DataFrame con nuestras notas médicas, nuestra expresión regular para la palabra a buscar, expresión regular para las expresiones como "historial familiar, no tiene historial de diabetes, no se ha identificado diabetes" entre otras y el tamaño de la ventana al rededor de la palabra a buscar.
## Función con expresiones regulares
```python 

def extract_text_window_pro(df, pattern,negatives, window_size, column_name = "TEXT"):
    
    #Constants
    half_window_size = window_size 
    final_df = pd.DataFrame([])
    column_position = df.columns.get_loc(column_name) + 1  #We add 1 cause position 0 is the index
    
    
    #Loop for each row of the column
    for row in df.itertuples():
        
        #Loop for multiple matches in the same row
        for match in re.finditer(pattern,row[column_position]):
            window_start = int([match.start()-half_window_size if match.start()>=half_window_size else 0][0])
            window_end = int([match.end() + half_window_size if match.end()+half_window_size <= len(row[column_position]) else len(row[column_position])][0])
            
            final_df = final_df.append({
                                        "WORD": match.group(),
                                        "START_INDEX": match.start(),
                                        "WINDOW_START": window_start,
                                        "WINDOW_END": window_end,
                                        "CONTEXT": row[column_position][window_start:window_end],
                                        "FULL_TEXT": row[column_position],
                                        "NOTE_ID": row[1]},
                                        ignore_index=True)
            #Extracción de negativos
            final_df2 = final_df[final_df["CONTEXT"].str.contains(pat = negatives, regex = True)==False]
    return "No matches for the pattern" if len(final_df) == 0 else  final_df2

# Buscando diabet en las notas médicas

df = pd.read_csv("https://raw.githubusercontent.com/hhsieh2416/Identify_Diabetic_Complications/main/data/diabetes_notes.csv")
pattern = "diabetes|diabetic" #"(?<![a-zA-Z])diabet(es|ic)?(?![a-zA-Z])"
window_size = 50
negatives = r"no history of (?<![a-zA-Z])diabet(es|ic)?(?![a-zA-z])|No history of (?<![a-zA-Z])diabet(es|ic)?(?![a-zA-z])|den(ies|y)? any comorbid complications|family history|negative for (?<![a-zA-Z])diabet(es|ic)?(?![a-zA-z])|(father|mother) (also)? (?<![a-zA-Z])diabet(es|ic)?(?![a-zA-z])|Negative for (?<![a-zA-Z])diabet(es|ic)?(?![a-zA-z]) |no weakness, numbness or tingling|patient's mother and father|father also has diabetes"
diabetes_notes_window = extract_text_window_pro(df,pattern,negatives,window_size)
diabetes_notes_window
```
Ahora bien, es momento de obtiene mediante la función con expresiones regulares los DataFrame para neuropathy, nephropathy y retinopathy.
```python
diabetes_notes_window.drop_duplicates(subset=["NOTE_ID"])
neuropathy = diabetes_notes_window[diabetes_notes_window['CONTEXT'].str.contains(pat=r"(?<![a-zA-Z])neuropath(y|ic)?(?![a-zA-z])|diabetic nerve pain|tingling",regex=True)]
neuropathy['COMPLICATIONS'] = "neuropathy"
diabetes_notes_neuropathy = neuropathy[['NOTE_ID','CONTEXT','COMPLICATIONS']].drop_duplicates(subset=['NOTE_ID'])
print(diabetes_notes_neuropathy)
print(diabetes_notes_neuropathy.count())
nephropathy =  diabetes_notes_window[diabetes_notes_window['CONTEXT'].str.contains(pat=r"(?<![a-zA-Z])nephropathy(?![a-zA-z])|renal (insufficiency|disease)",regex=True)]
nephropathy['COMPLICATIONS'] = "nephropathy"
diabetes_notes_nephropathy = nephropathy[['NOTE_ID','CONTEXT','COMPLICATIONS']].drop_duplicates(subset=['NOTE_ID'])
print(diabetes_notes_nephropathy)
print(diabetes_notes_nephropathy.count())
retinopathy = diabetes_notes_window[diabetes_notes_window['CONTEXT'].str.contains(pat=r"(?<![a-zA-Z])retinopath(y|ic)?(?![a-zA-z])",regex=True)]
retinopathy['COMPLICATIONS'] = "retinopathy"
diabetes_notes_retinopathy = retinopathy[['NOTE_ID','CONTEXT','COMPLICATIONS']].drop_duplicates(subset=['NOTE_ID'])
print(diabetes_notes_retinopathy)
print(diabetes_notes_retinopathy.count())
```
Para validar que nuestras funciones estén obteniendo bien la información de hace el uso del segundo link el cual se nos fue proporcionado para la validación de estas notas médicas.
```python
# Con el link antes mencionado de validación se crean los DataFrame para cada patología 
datos_verificacion = pd.read_csv("https://raw.githubusercontent.com/hhsieh2416/Identify_Diabetic_Complications/main/data/glodstandrad.csv")
datos_verificacion_neuropathy = datos_verificacion[datos_verificacion['DIABETIC_NEUROPATHY']==1][['NOTE_ID','DIABETIC_NEUROPATHY']]
print(datos_verificacion_neuropathy)
print(datos_verificacion_neuropathy.count())
datos_verificacion_nephropathy = datos_verificacion[datos_verificacion['DIABETIC_NEPHROPATHY']==1][['NOTE_ID','DIABETIC_NEPHROPATHY']]
print(datos_verificacion_nephropathy)
print(datos_verificacion_nephropathy.count())
datos_verificacion_retinopathy = datos_verificacion[datos_verificacion['DIABETIC_RETINOPATHY']==1][['NOTE_ID','DIABETIC_RETINOPATHY']]
print(datos_verificacion_retinopathy)
print(datos_verificacion_retinopathy.count())
# Realizamos joins de nuestros DataFrame con las tablas de validación
ver_neuro = pd.merge(datos_verificacion_neuropathy, diabetes_notes_neuropathy, how = 'outer', on = 'NOTE_ID', indicator=True)
print(ver_neuro)
ver_nephro = pd.merge(datos_verificacion_nephropathy, diabetes_notes_nephropathy, how = 'outer', on = 'NOTE_ID', indicator=True)
print(ver_nephro)
ver_retino = pd.merge(datos_verificacion_retinopathy, diabetes_notes_retinopathy, how = 'outer', on = 'NOTE_ID', indicator=True)
print(ver_retino)
# Se realizan los conteos
conteo_na_neuro_falso_positivo = ver_neuro['DIABETIC_NEUROPATHY'].isna().sum()
conteo_na_nephro_falso_positivo = ver_nephro['DIABETIC_NEPHROPATHY'].isna().sum()
conteo_na_retino_falso_positivo = ver_retino['DIABETIC_RETINOPATHY'].isna().sum()
print('Pacientes sin complicaciones pero que si se identifican: ', conteo_na_neuro_falso_positivo+conteo_na_nephro_falso_positivo+conteo_na_retino_falso_positivo)
conteo_na_neuro_falso_negativo = ver_neuro['COMPLICATIONS'].isna().sum()
conteo_na_nephro_falso_negativo = ver_nephro['COMPLICATIONS'].isna().sum()
conteo_na_retino_falso_negativo = ver_retino['COMPLICATIONS'].isna().sum()
print('Pacientes con complicaciones que no fueron detectados: ', conteo_na_neuro_falso_negativo + conteo_na_nephro_falso_negativo + conteo_na_retino_falso_negativo)
conteo_correcto_neuro = len(ver_neuro[ver_neuro['_merge'] == 'both'])
conteo_correcto_nephro = len(ver_nephro[ver_nephro['_merge'] == 'both'])
conteo_correcto_retino = len(ver_retino[ver_retino['_merge'] == 'both'])
print('Pacientes que tienen complicaciones diabetes que si se encontaron: ', conteo_correcto_nephro+conteo_correcto_neuro+conteo_correcto_retino)
conteo_complicacion_neuro = len( ver_neuro[ver_neuro['DIABETIC_NEUROPATHY'] == 1] )
conteo_complicacion_nephro = len( ver_nephro[ver_nephro['DIABETIC_NEPHROPATHY'] == 1] ) 
conteo_complicacion_retino = len( ver_retino[ver_retino['DIABETIC_RETINOPATHY'] == 1] ) 
print('Pacientes que tienen complicaciones diabeticas: ', conteo_complicacion_neuro +conteo_complicacion_nephro + conteo_complicacion_retino )
cor_neuro = datos_verificacion[['NOTE_ID', 'DIABETIC_NEUROPATHY']].merge(diabetes_notes_neuropathy[['NOTE_ID','COMPLICATIONS']], how='outer',  on='NOTE_ID', indicator=True )
cor_neuro['COMPLICATIONS'] = cor_neuro['COMPLICATIONS'].map(d_neuro).fillna(0)
print('---NEUROPATHY---')
print(cor_neuro)
print(classification_report(cor_neuro['DIABETIC_NEUROPATHY'].tolist(), cor_neuro['COMPLICATIONS'].tolist()))
cor_nephro = datos_verificacion[['NOTE_ID', 'DIABETIC_NEPHROPATHY']].merge(diabetes_notes_nephropathy[['NOTE_ID','COMPLICATIONS']], how='outer',  on='NOTE_ID', indicator=True )
cor_nephro['COMPLICATIONS'] = cor_nephro['COMPLICATIONS'].map(d_nephro).fillna(0)
print('---NEPHROPATHY---')
print(cor_nephro)
print(classification_report(cor_nephro['DIABETIC_NEPHROPATHY'].tolist(), cor_nephro['COMPLICATIONS'].tolist()))
cor_retino = datos_verificacion[['NOTE_ID', 'DIABETIC_RETINOPATHY']].merge(diabetes_notes_retinopathy[['NOTE_ID','COMPLICATIONS']], how='outer',  on='NOTE_ID', indicator=True )
cor_retino['COMPLICATIONS'] = cor_retino['COMPLICATIONS'].map(d_retino).fillna(0)
print('---RETINOPATHY---')
print(cor_retino)
print(classification_report(cor_retino['DIABETIC_RETINOPATHY'].tolist(), cor_retino['COMPLICATIONS'].tolist()))
```
