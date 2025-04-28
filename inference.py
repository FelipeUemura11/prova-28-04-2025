import pandas as pd
from pickle import load

def load_models():
    """
    Carrega os modelos salvos (scaler e kmeans)
    """
    try:
        scaler = load(open('./modelos/scaler_minmax.pkl', 'rb'))
        kmeans = load(open('./modelos/modelo_kmeans_clusters.pkl', 'rb'))
        # Carregar os dados normalizados originais para obter as colunas
        dados_normalizados = load(open('./modelos/dados_normalizados_minmax.pkl', 'rb'))
        return scaler, kmeans, dados_normalizados
    except Exception as e:
        raise Exception(f"Erro ao carregar os modelos: {str(e)}")

def preprocess_instance(instance, colunas_esperadas, scaler_modelo):
    """
    Preprocessa uma nova instancia para fazer a inferencia
    """
    # converte a instancia para DataFrame se nao for
    if not isinstance(instance, pd.DataFrame):
        instance = pd.DataFrame([instance])
    
    # separa as colunas categoricas e numericas
    colunas_categoricas = [
        '',
        '',
        '',
        '',
        ''
    ]
    
    colunas_numericas = [
        '',
        '',
        '',
        '',
        ''
    ]
    
    # Garantir que todas as colunas numéricas existam
    for col in colunas_numericas:
        if col not in instance.columns:
            instance[col] = 0
    
    # converte colunas numericas para float
    for coluna in colunas_numericas:
        instance[coluna] = instance[coluna].astype(float)
    
    # transforma as colunas categoricas em dummy variables
    colunas_categoricas_dummies = pd.get_dummies(instance[colunas_categoricas])
    
    # normaliza as colunas numericas usando o scaler do modelo
    colunas_numericas_normalizadas = pd.DataFrame(
        scaler_modelo.transform(instance[colunas_numericas]),
        columns=colunas_numericas
    )
    
    # concatena as colunas normalizadas e categoricas
    instance_processed = pd.concat([colunas_numericas_normalizadas, colunas_categoricas_dummies], axis=1)
    
    # Garantir que todas as colunas esperadas estejam presentes
    for coluna in colunas_esperadas:
        if coluna not in instance_processed.columns:
            instance_processed[coluna] = 0
    
    # Selecionar apenas as colunas esperadas na ordem correta
    instance_processed = instance_processed[colunas_esperadas]
    
    return instance_processed
    
def predict_cluster(instance):
    """
    Faz a predicaoo do cluster para uma nova instancia
    """
    try:
        # CARREGAMENTO DOS MODELOS
        scaler, kmeans, dados_normalizados = load_models()
        
        # Obter as colunas esperadas do modelo
        colunas_esperadas = dados_normalizados.columns.tolist()
        
        # PREPROCESSAMENTO DA INSTANCIA
        instance_processed = preprocess_instance(instance, colunas_esperadas, scaler)
        
        # Verificar se há diferenças entre as colunas
        colunas_faltando = set(colunas_esperadas) - set(instance_processed.columns)
        colunas_extras = set(instance_processed.columns) - set(colunas_esperadas)
        
        if colunas_faltando:
            print("\nColunas faltando:")
            print(list(colunas_faltando))
        
        if colunas_extras:
            print("\nColunas extras:")
            print(list(colunas_extras))
        
        # PREDICT
        cluster = kmeans.predict(instance_processed)
        
        return cluster[0], instance_processed
        
    except Exception as e:
        raise Exception(f"Erro durante a predição: {str(e)}")

def funcao_predict(instance):
    """
    Faz a predição do diagnóstico (Diagnostico_N ou Diagnostico_O) para uma nova instância
    """
    try:
        # CARREGAMENTO DOS MODELOS
        scaler, kmeans, dados_normalizados = load_models()
        
        # Obter as colunas esperadas do modelo
        colunas_esperadas = dados_normalizados.columns.tolist()
        
        # PREPROCESSAMENTO DA INSTANCIA
        instance_processed = preprocess_instance(instance, colunas_esperadas, scaler)
        
        # PREDICT CLUSTER
        cluster = kmeans.predict(instance_processed)
        
        # Mapear cluster para diagnóstico
        # Cluster 0 = predict1, Cluster 1 = predict2, etc.
        resultado = 'predict1' if cluster[0] == 0 else 'predict2'
        
        return resultado, cluster[0], instance_processed
        
    except Exception as e:
        raise Exception(f"Erro durante a predição do diagnóstico: {str(e)}")

if __name__ == "__main__":
    nova_instancia = { #
        '': '',
        '': 0,
        '': 0,
        '': '',
        '': 0
    }
    
    try:
        resultado, cluster, instancia_processada = funcao_predict(nova_instancia)
        
        print(f"\n >> Resultados da predicao << \n")
        print(f"predito: {resultado}")
        print(f"Cluster predito: {cluster}")
        print("\n >> Dados processados << \n")
        print(instancia_processada)

        # Imprimir informacoes adicionais sobre o diagnostico
        print(f"\n >> Informações do diagnóstico {resultado}: ")
        if resultado == 'predict1':
            print("Diagnóstico: Normal")
        else:
            print("Diagnóstico: Obesidade")
            
    except Exception as e:
        print(f"Erro: {str(e)}") 