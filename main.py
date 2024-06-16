# ..............................................................LIBRERÍAS..............................................
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# ...............................................................PRE PROCESAMIENTO DE DATOS................................
# Asegurarse de que las columnas de fecha y hora sean cadenas
df['FECHAINICIO'] = df['FECHAINICIO'].astype(str)
df['HORAINICIO'] = df['HORAINICIO'].astype(str)
df['FECHAFIN'] = df['FECHAFIN'].astype(str)
df['HORAFIN'] = df['HORAFIN'].astype(str)

# Concatenar y convertir a datetime
df['FECHAINICIO'] = pd.to_datetime(df['FECHAINICIO'] + ' ' + df['HORAINICIO'])
df['FECHAFIN'] = pd.to_datetime(df['FECHAFIN'] + ' ' + df['HORAFIN'])

# Seleccionar características relevantes para el clustering
features = df[['FECHAINICIO', 'FECHAFIN', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'NUMCONEXDOM', 'UNIDADESUSO']]

# Convertir columnas de fecha a valores numéricos (timestamp) para el clustering
features_numeric = features.copy()
features_numeric['FECHAINICIO'] = features_numeric['FECHAINICIO'].astype('int64') // 10**9
features_numeric['FECHAFIN'] = features_numeric['FECHAFIN'].astype('int64') // 10**9

# Convertir columnas categóricas a variables dummy
features_numeric = pd.get_dummies(features_numeric, columns=['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO'])

# Guardar el DataFrame preprocesado
features_numeric.to_csv('Interrupciones_Dataset_Preprocesado.csv', index=False)
# ...............................................................PRE PROCESAMIENTO DE DATOS................................

#####SEGUNDO COMMIT


# ...............................................................CLUSTERING................................................
# Realizar clustering con K-Means
num_clusters = 2  # Número de clusters, puedes ajustarlo según sea necesario
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(features_numeric)

# Añadir los resultados del clustering al DataFrame original
df['Cluster'] = kmeans.labels_

# Guardar el DataFrame con los resultados del clustering
df.to_csv('Interrupciones_Clusterizado.csv', index=False)
# ...............................................................CLUSTERING................................................


# ...........................................................GRÁFICOS DE DISPERSIÓN........................................
# FECHAINICIO vs FECHAFIN
plt.figure(figsize=(10, 6))
plt.scatter(df['FECHAINICIO'], df['FECHAFIN'], c=df['Cluster'], cmap='viridis')
plt.title('Grafico de dispersión de FECHAINICIO vs FECHAFIN por Cluster')
plt.xlabel('FECHAINICIO')
plt.ylabel('FECHAFIN')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# NUMCONEXDOM vs UNIDADESUSO
plt.figure(figsize=(10, 6))
plt.scatter(df['NUMCONEXDOM'], df['UNIDADESUSO'], c=df['Cluster'], cmap='viridis')
plt.title('Grafico de dispersión de NUMCONEXDOM vs UNIDADESUSO coloreado por Cluster')
plt.xlabel('NUMCONEXDOM')
plt.ylabel('UNIDADESUSO')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


# SFECHAINICIO vs UNIDADESUSO
plt.figure(figsize=(10, 6))
plt.scatter(df['FECHAINICIO'], df['UNIDADESUSO'], c=df['Cluster'], cmap='viridis')
plt.title('Grafico de dispersión de FECHAINICIO vs UNIDADESUSO coloreado por Cluster')
plt.xlabel('FECHAINICIO')
plt.ylabel('UNIDADESUSO')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
# ...........................................................GRÁFICOS DE DISPERSIÓN........................................

#####TERCER COMMIT

# ...............................................................DATOS ESTADISTICOS PARTE 1................................
# 1. Cantidad de interrupciones por motivo
interrupciones_por_motivo = df['MOTIVOINTERRUPCION'].value_counts()

# Grafico de barras usando Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x=interrupciones_por_motivo.index, y=interrupciones_por_motivo.values, palette='viridis')

# Etiquetas y titulo
plt.xlabel('Motivo de interrupción')
plt.ylabel('Cantidad de interrupciones')
plt.title('Cantidad de interrupciones por motivo')
plt.xticks(rotation=90)
plt.xticks(rotation=45, ha='right')  # para mejor lectura
plt.tight_layout()
plt.show()

# 2. Cantidad de interrupciones por departamento
interrupciones_por_departamento = df['DEPARTAMENTO'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=interrupciones_por_departamento.index, y=interrupciones_por_departamento.values, palette='viridis')

plt.xlabel('Departamento')
plt.ylabel('Cantidad de interrupciones')
plt.title('Cantidad de interrupciones por departamento')
plt.xticks(rotation=45, ha='right')  # para mejor lectura
plt.tight_layout()
plt.show()

# 3. Cantidad de interrupciones por mes
df['MESINICIO'] = df['FECHAINICIO'].dt.month # Extraer el mes de la fecha de inicio

interrupciones_por_mes = df['MESINICIO'].value_counts().sort_index() #se ordenan los meses

plt.figure(figsize=(10, 6))
sns.barplot(x=interrupciones_por_mes.index, y=interrupciones_por_mes.values, palette='viridis')

plt.xlabel('Mes')
plt.ylabel('Cantidad de interrupciones')
plt.title('Cantidad de interrupciones por mes')
plt.xticks(rotation=45, ha='right')  # para mejor lectura
plt.tight_layout()
plt.show()

#####CUARTO COMMIT
