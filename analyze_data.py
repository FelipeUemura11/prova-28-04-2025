import pandas as pd

# Lendo os dados
print("Lendo os dados...")
df = pd.read_csv('diabetic_data.csv')

# Informações básicas sobre o conjunto de dados
print("\nFormato do Dataset:", df.shape)
print("\nColunas:", df.columns.tolist())

# Analisando cada coluna
print("\nAnalisando colunas...")
for column in df.columns:
    # Calculando valores faltantes
    missing = df[column].isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    # Calculando valores únicos
    unique_values = df[column].nunique()
    
    # Calculando valor mais comum e sua frequência
    most_common = df[column].mode()[0]
    most_common_count = (df[column] == most_common).sum()
    most_common_percent = (most_common_count / len(df)) * 100
    
    print(f"\nColuna: {column}")
    print(f"Valores faltantes: {missing} ({missing_percent:.2f}%)")
    print(f"Valores únicos: {unique_values}")
    print(f"Valor mais comum: {most_common} ({most_common_percent:.2f}%)")
    
    # Verificando se a coluna é constante (todos os valores são iguais)
    if unique_values == 1:
        print("AVISO: Esta coluna é constante - pode ser removida")
    
    # Verificando se a coluna tem muitos valores faltantes (>50%)
    if missing_percent > 50:
        print("AVISO: Esta coluna tem muitos valores faltantes - pode ser removida")
    
    # Verificando se a coluna tem muitos valores únicos em relação ao total de linhas (>90%)
    if unique_values > 0.9 * len(df):
        print("AVISO: Esta coluna tem muitos valores únicos - pode ser um ID ou timestamp")

# Salvando os resultados da análise
with open('column_analysis.txt', 'w') as f:
    f.write("Resultados da Análise de Colunas\n")
    f.write("==============================\n\n")
    for column in df.columns:
        f.write(f"Coluna: {column}\n")
        f.write(f"Valores faltantes: {df[column].isnull().sum()} ({(df[column].isnull().sum() / len(df)) * 100:.2f}%)\n")
        f.write(f"Valores únicos: {df[column].nunique()}\n")
        f.write(f"Valor mais comum: {df[column].mode()[0]} ({(df[column] == df[column].mode()[0]).sum() / len(df) * 100:.2f}%)\n")
        f.write("-------------------\n")

print("\nAnálise completa! Resultados foram salvos em 'column_analysis.txt'") 