#%%
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split

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
df_ajustado = df.copy()
df_ajustado['sex'] = label_encoder.fit_transform(df['sex'])
df_ajustado['smoker'] = label_encoder.fit_transform(df['smoker'])
df_ajustado = pd.get_dummies(df_ajustado)

#%%
# =============================================================================
# ANÁLISE DOS DADOS COM GRÁFICOS
# =============================================================================
# Histogramas de frequencia dos dados
df_ajustado.hist(bins=50, figsize=(20, 15))
plt.show()

# Matriz de correlação
corr_matrix = df_ajustado.corr(numeric_only=True)
print(
    corr_matrix["charges"].sort_values(ascending=False)
)

# Heat Map correlação
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    df_ajustado.corr(),
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    vmin=-1.0, vmax=1.0,
    square=True,
    ax=ax
)
plt.show()

plt.boxplot(df_ajustado['age'])
plt.title('age')
plt.ylabel('Valores')
plt.show()
plt.boxplot(df_ajustado['bmi'])
plt.title('bmi')
plt.ylabel('Valores')
plt.show()
plt.boxplot(df_ajustado['smoker'])
plt.title('smoker')
plt.ylabel('Valores')
plt.show()
plt.boxplot(df_ajustado['charges'])
plt.title('charges')
plt.ylabel('Valores')
plt.show()


#%%
# =============================================================================
# PREPARAÇÃO DOS DADOS
# =============================================================================









#%%
# =============================================================================
# SEPARAÇÃO DOS DADOS
# =============================================================================

y = df_ajustado["charges"]
X = df_ajustado.copy().drop("charges", axis = 1)
X.drop("sex", axis = 1, inplace=True)
X.drop("children", axis = 1, inplace=True)
X.drop("region_northwest", axis = 1, inplace=True)
X.drop("region_southeast", axis = 1, inplace=True)
X.drop("region_southwest", axis = 1, inplace=True)
X.drop("region_northeast", axis = 1, inplace=True)
# y.head()
# X.head()

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)
print('\n', len(X_train), 'treinamento +', len(X_test), 'teste\n')

















