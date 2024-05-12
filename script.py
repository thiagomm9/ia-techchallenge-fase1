#%%
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('base.csv')

#%%
# =============================================================================
# ANÁLISE INICIAL DA BASE
# =============================================================================
print('Verificando as colunas:')
df.head()

#%%
print('Verificando se há valores nulos:')
df.isnull().sum()

#%%
print('Tamanho da base:')
df.shape

#%%
print('Informações dos dados:')
df.info()

#%%
print('Números da base:')
df.describe()

#%%
print('Quantidade de valores únicos na coluna region:')
df['region'].nunique()

#%%
# =============================================================================
# TRANSFORMAÇÕES E AJUSTES DOS DADOS
# =============================================================================
label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])
df['smoker'] = label_encoder.fit_transform(df['smoker'])
df_ajustado = pd.get_dummies(df)

#%%
# =============================================================================
# ANÁLISE DOS DADOS COM GRÁFICOS
# =============================================================================
df_ajustado.hist(bins=50, figsize=(20, 15))
plt.show()