import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import os

# Criando diretório modelos se não existir
modelos_dir = 'modelos'
if not os.path.exists(modelos_dir):
    os.makedirs(modelos_dir)
    print(f"\n >> Diretório '{modelos_dir}' criado com sucesso <<")

try:
    # Gerando dados de exemplo
    print("\n >> Lendo conjunto de dados << \n")
    df = pd.read_csv('./nome_arquivo', sep=',')

    print(f"Shape original: {df.shape}")

    # eliminando os valores nulos
    df = df.dropna()

    # Definindo as colunas >> categoricas e numericas <<
    colunas_categoricas = df[[
        '',
        '',
        '',
        '',
        ''
    ]]
    colunas_numericas = df.drop(columns=[
        '',
        '',
        '',
        '',
        '',
        ''
    ])

    # transformar os dados categoricos em binario para a leitura do computador
    colunas_categoricas_dummies = pd.get_dummies(colunas_categoricas)

    # convertendo todas as colunas numericas para float
    colunas_numericas = colunas_numericas.copy()
    for c in colunas_numericas.columns:
        colunas_numericas[c] = pd.to_numeric(colunas_numericas[c], errors='coerce')

    # normalizando os dados
    print("\n >> Normalizando os dados com MinMaxScaler << ")
    scaler = MinMaxScaler()
    colunas_numericas_normalizadas = pd.DataFrame(
        scaler.fit_transform(colunas_numericas),
        columns=colunas_numericas.columns
    )

    # concatenando as colunas categoricas e numericas
    df_normalizado = pd.concat([colunas_numericas_normalizadas, colunas_categoricas_dummies], axis=1)
    
    # Convertendo para array numpy para clustering
    X = df_normalizado.values.astype(np.float64)

    # salvando os dados normalizados e o scaler
    print("\n >> Salvando dados normalizados <<")
    dump(df_normalizado, open(os.path.join(modelos_dir, 'dados_normalizados_minmax.pkl'), 'wb'))
    dump(scaler, open(os.path.join(modelos_dir, 'scaler_minmax.pkl'), 'wb'))
    print("    >Dados salvos com sucesso<    ")

    # analise de clustering
    print("\n >> Realizando analise de clustering <<")
    distortions = []
    K_range = range(1, X.shape[0] + 1)

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

    #desterminar o numero otimo de clustes
    #Metodo do cotovelo
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

    plt.figure(figsize=(12, 8))
    plt.plot(K_range, distortions, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Numero de Clusters (K)', fontsize=12)
    plt.ylabel('Inercia', fontsize=12)
    plt.title('Metodo do Cotovelo para Determinacao do Numero Otimo de Clusters', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.axvline(x=numero_clusters_otimo, color='r', linestyle='--', label=f'Clusters escolhido: {numero_clusters_otimo}')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

    # aplicando K-means com 2 clusters pois o contexto quer dois grupos
    # (pode ser alterado para o numero de clusters otimo encontrado acima)
    print("\n >> Aplicando K-means com k=2 <<")
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # salvando o modelo K-means completo e os rotulos
    print("\n >> Salvando modelo K-means e rotulos <<")
    try:
        # verificar se o diretorio modelos tem permissao de escrita
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
    raise e  # Adicionando raise para ver o traceback completo