# ..............................................................LIBRERÍAS..............................................
import pandas as pd

# ..............................................................LIBRERÍAS..............................................


# ...............................................................CARGA INICIAL DE DATOS................................
df = pd.read_csv('Interrupciones_Dataset.csv') 
# ...............................................................CARGA INICIAL DE DATOS................................


# ...............................................................LIMPIEZA DE DATOS.....................................
# Eliminamos la primera columna que es irrelevante
df = df.iloc[:, 1:]

# Eliminar los registros con valores nulos en la columna NUMCONEXDOM
columnas_valores_vacios = 'NUMCONEXDOM' 
df = df.dropna(subset=[columnas_valores_vacios])

# Reemplazar los valores nulos por cero en NUMCAMIONESPUNTOS
camiones_apoyo_valores_nulos = 'NUMCAMIONESPUNTOS'
df[camiones_apoyo_valores_nulos] = df[camiones_apoyo_valores_nulos].fillna(0).replace(0, 0)
# ...............................................................LIMPIEZA DE DATOS.........................................

#####PRIMER COMMIT
