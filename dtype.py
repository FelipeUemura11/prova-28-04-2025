import pandas as pd

df = pd.read_csv('./diabetic_data.csv', sep=',')

# para ver os tipos de dados antes de normalizar
print(df.dtypes)
