import pandas as pd
import numpy as np
from pickle import load
import os

def carregar_modelos():
    """Carrega os modelos salvos do diretório 'modelos'"""
    modelos_dir = 'modelos'
    
    try:
        # Carregando os modelos
        dados_normalizados = load(open(os.path.join(modelos_dir, 'dados_normalizados_minmax.pkl'), 'rb'))
        kmeans = load(open(os.path.join(modelos_dir, 'modelo_kmeans_clusters.pkl'), 'rb'))
        scaler = load(open(os.path.join(modelos_dir, 'scaler_minmax.pkl'), 'rb'))
        
        print("\n>> Modelos carregados com sucesso <<")
        print(f"\nNúmero de features esperado pelo modelo: {kmeans.n_features_in_}")
        return dados_normalizados, kmeans, scaler
    except Exception as e:
        print(f"Erro ao carregar modelos: {str(e)}")
        raise e

def criar_instancia_exemplo():
    """Cria uma instância de exemplo para inferência"""
    print("\n>> Criando instância de exemplo <<")
    
    # Dados numéricos
    dados_numericos = {
        'time_in_hospital': 5.0,
        'num_lab_procedures': 45.0,
        'num_procedures': 2.0,
        'num_medications': 15.0,
        'number_outpatient': 2.0,
        'number_emergency': 1.0,
        'number_inpatient': 1.0,
        'number_diagnoses': 9.0
    }
    
    # Dados categóricos
    dados_categoricos = {
        'race': 'Caucasian',
        'gender': 'Female',
        'age': '[40-50)',
        'admission_type_id': 1,
        'discharge_disposition_id': 1,
        'admission_source_id': 7,
        'metformin': 'Steady',
        'repaglinide': 'No',
        'nateglinide': 'No',
        'chlorpropamide': 'No',
        'glimepiride': 'No',
        'glipizide': 'No',
        'glyburide': 'No',
        'pioglitazone': 'No',
        'rosiglitazone': 'No',
        'acarbose': 'No',
        'miglitol': 'No',
        'tolazamide': 'No',
        'insulin': 'Steady',
        'glyburide-metformin': 'No',
        'glipizide-metformin': 'No',
        'change': 'No',
        'diabetesMed': 'Yes',
        'readmitted': 'NO'
    }
    
    print(">Instância de exemplo criada com sucesso<")
    return dados_numericos, dados_categoricos

# Definindo as colunas numéricas e categóricas
colunas_numericas = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses'
]

colunas_categoricas = [
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
]

# Definindo todos os valores possíveis para cada variável categórica
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

def processar_instancia(dados_numericos, dados_categoricos, scaler):
    """Processa a nova instância para inferência"""
    # Criar DataFrames com as colunas corretas
    df_numericos = pd.DataFrame([dados_numericos])
    df_categoricos = pd.DataFrame([dados_categoricos])
    
    # Substituir '?' por np.nan
    df_numericos = df_numericos.replace('?', np.nan)
    df_categoricos = df_categoricos.replace('?', np.nan)
    
    # Preencher valores ausentes com a média para colunas numéricas
    for coluna in df_numericos.columns:
        media = df_numericos[coluna].mean()
        df_numericos[coluna] = df_numericos[coluna].fillna(media)
    
    # Preencher valores ausentes com a moda para colunas categóricas
    for coluna in df_categoricos.columns:
        valores = df_categoricos[coluna].value_counts()
        if not valores.empty:
            moda = valores.index[0]
            df_categoricos[coluna] = df_categoricos[coluna].fillna(moda)
    
    # Normalizar dados numéricos
    df_numericos_norm = pd.DataFrame(
        scaler.transform(df_numericos),
        columns=df_numericos.columns
    )
    
    # Criar dummy variables para colunas categóricas
    df_categoricos_dummies = pd.DataFrame()
    for coluna in df_categoricos.columns:
        # Criar dummies para a coluna atual
        dummies = pd.get_dummies(df_categoricos[coluna], prefix=coluna)
        
        # Adicionar colunas faltantes com zeros
        for valor in valores_categoricos[coluna]:
            coluna_dummy = f"{coluna}_{valor}"
            if coluna_dummy not in dummies.columns:
                dummies[coluna_dummy] = 0
        
        # Concatenar com o DataFrame de dummies
        df_categoricos_dummies = pd.concat([df_categoricos_dummies, dummies], axis=1)
    
    # Concatenar dados normalizados
    dados_processados = pd.concat([df_numericos_norm, df_categoricos_dummies], axis=1)
    
    # Preencher valores NaN restantes com 0
    dados_processados = dados_processados.fillna(0)
    
    print(f"\nNúmero de features após processamento: {dados_processados.shape[1]}")
    
    return dados_processados

def inferir_cluster(dados_processados, kmeans, dados_normalizados):
    """
    Infere o cluster para uma nova instância e retorna os dados desnormalizados do centroide
    """
    try:
        # Fazer a predição
        cluster = kmeans.predict(dados_processados)[0]
        
        # Calcular a distância para o centroide mais próximo
        distancia = kmeans.transform(dados_processados).min(axis=1)[0]
        
        # Obter o centroide do cluster
        centroide = kmeans.cluster_centers_[cluster]
        
        # Calcular estatísticas básicas do cluster
        estatisticas = {
            'cluster': cluster,
            'distancia_centroide': distancia,
            'numero_features': dados_processados.shape[1],
            'centroide': centroide
        }
        
        return cluster, distancia, estatisticas
    except Exception as e:
        print(f"Erro ao inferir cluster: {str(e)}")
        raise e

def exibir_resultados(cluster, distancia, estatisticas, dados_numericos, dados_categoricos, scaler):
    """Exibe os resultados da inferência de forma simplificada"""
    print("\n>> Resultados da Inferência <<")
    print(f"\nCluster: {estatisticas['cluster']}")
    print(f"Distância do centroide: {estatisticas['distancia_centroide']:.2f}")
    
    # Desnormalizar os dados numéricos
    dados_numericos_array = np.array([[
        dados_numericos['time_in_hospital'],
        dados_numericos['num_lab_procedures'],
        dados_numericos['num_procedures'],
        dados_numericos['num_medications'],
        dados_numericos['number_outpatient'],
        dados_numericos['number_emergency'],
        dados_numericos['number_inpatient'],
        dados_numericos['number_diagnoses']
    ]])
    
    dados_desnormalizados = scaler.inverse_transform(dados_numericos_array)[0]
    
    print("\nDados do Paciente:")
    print(f"Tempo no hospital: {int(dados_desnormalizados[0])} dias")
    print(f"Número de procedimentos: {int(dados_desnormalizados[2])}")
    print(f"Número de medicamentos: {int(dados_desnormalizados[3])}")
    
    print("\nPerfil:")
    print(f"Gênero: {dados_categoricos['gender']}")
    print(f"Faixa etária: {dados_categoricos['age']}")
    print(f"Uso de insulina: {dados_categoricos['insulin']}")
    print(f"Readmissão: {dados_categoricos['readmitted']}")
    
    # Interpretação simplificada
    print("\nStatus:", end=" ")
    if distancia < 2:
        print("Perfil típico do cluster")
    elif distancia < 4:
        print("Perfil comum do cluster")
    else:
        print("Perfil único - requer atenção especial")

def main():
    try:
        # Carregando modelos
        dados_normalizados, kmeans, scaler = carregar_modelos()
        
        # Criando instância de exemplo
        dados_numericos, dados_categoricos = criar_instancia_exemplo()
        
        # Processando instância
        dados_processados = processar_instancia(dados_numericos, dados_categoricos, scaler)
        
        # Inferindo cluster
        cluster, distancia, estatisticas = inferir_cluster(dados_processados, kmeans, dados_normalizados)
        
        # Exibindo resultados
        exibir_resultados(cluster, distancia, estatisticas, dados_numericos, dados_categoricos, scaler)
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 