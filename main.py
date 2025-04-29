# Feito por Felipe Yukiya Soares Uemura
# 2025-04-28

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import os

# criando diretorio modelos se nao existir
modelos_dir = 'modelos'
if not os.path.exists(modelos_dir):
    os.makedirs(modelos_dir)
    print(f"\n >> Diretório '{modelos_dir}' criado com sucesso <<")

try:
    # lendo conjunto de dados
    print("\n >> Lendo conjunto de dados << \n")
    df_original = pd.read_csv('./diabetic_data.csv', sep=',')

    print(f"Shape original: {df_original.shape}")

    # realizando amostragem de 10% dos dados para analise do cotovelo
    df = df_original.sample(frac=0.1, random_state=42)
    print(f"Shape apos amostragem: {df.shape}")

    # eliminando os valores nulos
    df = df.dropna()

    # removendo colunas problematicas
    # Elas foram encontradas no arquivo analyze_data.py
    print("\n >> removendo colunas problematica <<")
    colunas_para_remover = [
        'encounter_id',  # identificador unico
        'patient_nbr',   # identificador unico
        'weight',        # muitos valores ausentes
        'payer_code',    # muitos valores ausentes
        'medical_specialty',  # muitos valores ausentes
        'diag_1',        # muitos valores unico
        'diag_2',        # muitos valores unico
        'diag_3',        # muitos valores unico
        'max_glu_serum', # muitos valores ausentes
        'A1Cresult',     # muitos valores ausentes
        'acetohexamide', # valor constante
        'tolbutamide',   # valor constante
        'troglitazone',  # valor constante
        'examide',       # valor constante
        'citoglipton',   # valor constante
        'glimepiride-pioglitazone',  # valor constante
        'metformin-rosiglitazone',   # valor constante
        'metformin-pioglitazone'     # valor constante
    ]
    
    df = df.drop(columns=colunas_para_remover)
    print(f"  >Removidas {len(colunas_para_remover)} colunas problematica<")

    # definindo as colunas >> categoricas e numericas <<
    colunas_categoricas = df[[
        'race',
        'gender',
        'age',
        'admission_type_id',
        'discharge_disposition_id',
        'admission_source_id',
        'metformin',
        'repaglinide',
        'nateglinide',
        'chlorpropamide',
        'glimepiride',
        'glipizide',
        'glyburide',
        'pioglitazone',
        'rosiglitazone',
        'acarbose',
        'miglitol',
        'tolazamide',
        'insulin',
        'glyburide-metformin',
        'glipizide-metformin',
        'change',
        'diabetesMed',
        'readmitted'
    ]]
    colunas_numericas = df[[
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_outpatient',
        'number_emergency',
        'number_inpatient',
        'number_diagnoses'
    ]]

    # Definindo todos os valores possíveis para cada variável categórica
    # garantir que, ao criar variáveis dummy (one-hot encoding), todas as categorias possiveis sejam consideradas, mesmo que nao estejam presentes na amostra atual.
    valores_categoricos = {
        'race': ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'],
        'gender': ['Male', 'Female'],
        'age': ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)'],
        'admission_type_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'discharge_disposition_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        'admission_source_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        'metformin': ['Up', 'Down', 'Steady', 'No'],
        'repaglinide': ['Up', 'Down', 'Steady', 'No'],
        'nateglinide': ['Up', 'Down', 'Steady', 'No'],
        'chlorpropamide': ['Up', 'Down', 'Steady', 'No'],
        'glimepiride': ['Up', 'Down', 'Steady', 'No'],
        'glipizide': ['Up', 'Down', 'Steady', 'No'],
        'glyburide': ['Up', 'Down', 'Steady', 'No'],
        'pioglitazone': ['Up', 'Down', 'Steady', 'No'],
        'rosiglitazone': ['Up', 'Down', 'Steady', 'No'],
        'acarbose': ['Up', 'Down', 'Steady', 'No'],
        'miglitol': ['Up', 'Down', 'Steady', 'No'],
        'tolazamide': ['Up', 'Down', 'Steady', 'No'],
        'insulin': ['Up', 'Down', 'Steady', 'No'],
        'glyburide-metformin': ['Up', 'Down', 'Steady', 'No'],
        'glipizide-metformin': ['Up', 'Down', 'Steady', 'No'],
        'change': ['Ch', 'No'],
        'diabetesMed': ['Yes', 'No'],
        'readmitted': ['<30', '>30', 'NO']
    }

    # substituindo '?' por np.nan nas colunas numericas/categoricas
    for coluna in colunas_numericas.columns:
        colunas_numericas.loc[:, coluna] = colunas_numericas[coluna].replace('?', np.nan)
    for coluna in colunas_categoricas.columns:
        colunas_categoricas.loc[:, coluna] = colunas_categoricas[coluna].replace('?', np.nan)

    # preenchendo valores ausentes com a media para colunas numericas
    for coluna in colunas_numericas.columns:
        media = colunas_numericas[coluna].mean()
        colunas_numericas.loc[:, coluna] = colunas_numericas[coluna].fillna(media)

    # preenchendo valores ausentes com moda para colunas categoricas
    for coluna in colunas_categoricas.columns:
        valores = colunas_categoricas[coluna].value_counts()
        if not valores.empty:
            moda = valores.index[0]
            colunas_categoricas.loc[:, coluna] = colunas_categoricas[coluna].fillna(moda)
        else:
            print(f"  >Coluna '{coluna}' não tem valores para calcular moda<")

    # Criando dummy variables para cada coluna categorica com todos os valores possiveis
    # para transformar em binarios
    colunas_categoricas_dummies = pd.DataFrame()
    for coluna in colunas_categoricas.columns:

        dummies = pd.get_dummies(colunas_categoricas[coluna], prefix=coluna)
        
        # Adicionando colunas faltantes com zeros
        for valor in valores_categoricos[coluna]:
            coluna_dummy = f"{coluna}_{valor}"
            if coluna_dummy not in dummies.columns:
                dummies[coluna_dummy] = 0
        
        # Concatenando com o DataFrame de dummy variables
        colunas_categoricas_dummies = pd.concat([colunas_categoricas_dummies, dummies], axis=1)

    # convertendo todas as colunas numericas para float
    colunas_numericas = colunas_numericas.astype(float)

    # normalizando os dados
    print("\n >> Normalizando os dados com MinMaxScaler << ")
    scaler = MinMaxScaler()
    colunas_numericas_normalizadas = pd.DataFrame(
        scaler.fit_transform(colunas_numericas),
        columns=colunas_numericas.columns
    )

    # concatenando as colunas categoricas e numericas
    df_normalizado = pd.concat([colunas_numericas_normalizadas, colunas_categoricas_dummies], axis=1)
    
    # Preenchendo valores NaN restantes com 0
    df_normalizado = df_normalizado.fillna(0)
    
    # Convertendo para array numpy para clustering
    X = df_normalizado.values.astype(np.float64)

    # Verificando valores NaN antes do clustering
    print("\n >> Verificando valores NaN antes do clustering <<")
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"Encontrados {nan_count} valores NaN no array X")
        print("\nColunas com NaN:")
        for col in df_normalizado.columns[df_normalizado.isna().any()]:
            print(f"- {col}: {df_normalizado[col].isna().sum()} valores NaN")

    # salvando os dados normalizados e o scaler
    print("\n >> Salvando dados normalizados <<")
    dump(df_normalizado, open(os.path.join(modelos_dir, 'dados_normalizados_minmax.pkl'), 'wb'))
    dump(scaler, open(os.path.join(modelos_dir, 'scaler_minmax.pkl'), 'wb'))
    print("    >Dados salvos com sucesso<    ")

    # analise de clustering
    print("\n >> Realizando analise de clustering <<")
    distortions = []
    #K_range = range(1, df_normalizado.shape[0] + 1)
    K_range = range(1, 51)  # testando 50 clusters

    # Aula 3 (fertility)
    #determinar o numero otimo de clustes
    for k in K_range:
        cluster_modelo = KMeans(n_clusters=k, random_state=42).fit(X)
        #calcular a distocao obtida com o modelo treinado de k centroides
        distortions.append(
            sum(
                np.min(
                    cdist(X, cluster_modelo.cluster_centers_, 'euclidean'), axis=1
                )/X.shape[0]
            )
        )
    x0 = K_range[0]
    y0 = distortions[0]
    x1 = K_range[-1]
    y1 = distortions[-1]

    distancias = []
    for i in range(len(distortions)):
        x = K_range[i]
        y = distortions[i]
        distancia = abs((y1-y0)/(x1-x0)*(x-x0)+y0-y)
        distancias.append(distancia)

    numero_clusters_otimo = distancias.index(max(distancias)) + 1
    print(f"\nNumero otimo de clusters: {numero_clusters_otimo}")

    #Metodo do cotovelo
    plt.figure(figsize=(15, 10))
    plt.plot(K_range, distortions, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Numero de Clusters (K)', fontsize=12)
    plt.ylabel('Inercia', fontsize=12)
    plt.title('Metodo do Cotovelo para Determinacao do Numero Otimo de Clusters', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.axvline(x=numero_clusters_otimo, color='r', linestyle='--', label=f'Clusters escolhido: {numero_clusters_otimo}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('elbow_method.png')  # Salvando o gráfico
    plt.show()

    # aplicando K-means com o número ótimo de clusters encontrado na amostra
    print(f"\n >> Aplicando K-means com k={numero_clusters_otimo} na base completa <<")
    
    # Preparando a base original para o treinamento final
    df_original = df_original.dropna()
    colunas_categoricas_original = df_original[[
        'race',
        'gender',
        'age',
        'admission_type_id',
        'discharge_disposition_id',
        'admission_source_id',
        'metformin',
        'repaglinide',
        'nateglinide',
        'chlorpropamide',
        'glimepiride',
        'glipizide',
        'glyburide',
        'pioglitazone',
        'rosiglitazone',
        'acarbose',
        'miglitol',
        'tolazamide',
        'insulin',
        'glyburide-metformin',
        'glipizide-metformin',
        'change',
        'diabetesMed',
        'readmitted'
    ]]
    colunas_numericas_original = df_original[[
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_outpatient',
        'number_emergency',
        'number_inpatient',
        'number_diagnoses'
    ]]
    
    # substituindo '?' por NaN
    for coluna in colunas_numericas_original.columns:
        colunas_numericas_original.loc[:, coluna] = colunas_numericas_original[coluna].replace('?', np.nan)
    
    # substituindo '?' por NaN nas colunas categoricas da base original
    for coluna in colunas_categoricas_original.columns:
        colunas_categoricas_original.loc[:, coluna] = colunas_categoricas_original[coluna].replace('?', np.nan)
    
    # preenchendo valores ausentes com media para colunas numericas da base original
    for coluna in colunas_numericas_original.columns:
        media = colunas_numericas_original[coluna].mean()
        colunas_numericas_original.loc[:, coluna] = colunas_numericas_original[coluna].fillna(media)
    
    # preenchendo valores ausentes com moda para colunas categoricas da base original
    for coluna in colunas_categoricas_original.columns:
        valores = colunas_categoricas_original[coluna].value_counts()
        if not valores.empty:
            moda = valores.index[0]
            colunas_categoricas_original.loc[:, coluna] = colunas_categoricas_original[coluna].fillna(moda)
        else:
            print(f"    >Coluna '{coluna}' não tem valores para calcular moda<")
    
    # Criando dummy variables para cada coluna categórica da base original com todos os valores possíveis
    colunas_categoricas_dummies_original = pd.DataFrame()
    for coluna in colunas_categoricas_original.columns:
        # Criando dummy variables para a coluna atual
        dummies = pd.get_dummies(colunas_categoricas_original[coluna], prefix=coluna)
        
        # Adicionando colunas faltantes com zeros
        for valor in valores_categoricos[coluna]:
            coluna_dummy = f"{coluna}_{valor}"
            if coluna_dummy not in dummies.columns:
                dummies[coluna_dummy] = 0
        
        # Concatenando com o DataFrame de dummy variables
        colunas_categoricas_dummies_original = pd.concat([colunas_categoricas_dummies_original, dummies], axis=1)
    
    # convertendo colunas numericas da base original para float
    colunas_numericas_original = colunas_numericas_original.astype(float)
    
    # normalizando dados da base original
    colunas_numericas_normalizadas_original = pd.DataFrame(
        scaler.transform(colunas_numericas_original),
        columns=colunas_numericas_original.columns
    )
    
    # concatenando dados normalizados da base original
    df_normalizado_original = pd.concat([colunas_numericas_normalizadas_original, colunas_categoricas_dummies_original], axis=1)
    
    # preenchendo valores NaN restantes com 0 (assumindo que NaN em one-hot significa ausência da categoria)
    df_normalizado_original = df_normalizado_original.fillna(0)
    
    X_original = df_normalizado_original.values.astype(np.float64)
    
    # treinando K-means com a base completa
    kmeans = KMeans(n_clusters=numero_clusters_otimo, random_state=42)
    kmeans.fit(X_original)
    labels = kmeans.labels_

    # salvando o modelo K-means completo e os rotulos
    print("\n >> Salvando modelo K-means e rotulos <<")
    try:
        print(f"Diretorio atual: {os.getcwd()}")
        print(f"Permissao de escrita no diretorio: {os.access(modelos_dir, os.W_OK)}")
        
        # salvar o modelo K-means completo
        with open(os.path.join(modelos_dir, 'modelo_kmeans_clusters.pkl'), 'wb') as f:
            dump(kmeans, f)
        print("Modelo K-means completo salvo com sucesso!")
        
        # salvar os rotulos
        with open(os.path.join(modelos_dir, 'kmeans_labels_clusters.pkl'), 'wb') as f:
            dump(labels, f)
        print("Rotulos salvos com sucesso!")
            
    except Exception as e:
        print(f"Erro ao salvar modelo K-means ou rotulos: {str(e)}")

except Exception as e:
    print(f"Ocorreu um erro: {str(e)}")
    raise e  # adicionando raise para ver o traceback completo