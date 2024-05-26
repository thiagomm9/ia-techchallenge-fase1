#%%
# =============================================================================
# IMPORTS
# =============================================================================
import helpers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import GammaRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


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
# SEPARAÇÃO E PREPARAÇÃO DOS DADOS
# =============================================================================
# Remover outliers
# df_ajustado.drop(df_ajustado[df_ajustado.charges > 50000].index, inplace=True)
# df_ajustado.drop(df_ajustado[df_ajustado.bmi > 50].index, inplace=True)
# Sem melhoras significativas com a remoção de outliers

y = df_ajustado["charges"]
X = df_ajustado.copy().drop("charges", axis = 1)
X.drop("sex", axis = 1, inplace=True)
X.drop("children", axis = 1, inplace=True)
X.drop("region_northwest", axis = 1, inplace=True)
X.drop("region_southeast", axis = 1, inplace=True)
X.drop("region_southwest", axis = 1, inplace=True)
X.drop("region_northeast", axis = 1, inplace=True)
y.head()
X.head()

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)
print('\n', len(X_train), 'treinamento +', len(X_test), 'teste\n')

#%%
# =============================================================================
# ESCALONAMENTO DOS DADOS
# =============================================================================
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_min_max_scaled = scaler.transform(X_train)
X_test_min_max_scaled = scaler.transform(X_test)
scaler.fit(X)
X_min_max_scaled = scaler.transform(X)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_standard_scaled = scaler.transform(X_train)
X_test_standard_scaled = scaler.transform(X_test)
scaler.fit(X)
X_standard_scaled = scaler.transform(X)

#%%
# =============================================================================
# TREINA E AVALIA MODELO COM REGRESSÃO LINEAR
# =============================================================================
# Não há diferenças relevantes com o escalonamento dos dados
model_lin_reg = LinearRegression()
model_lin_reg.fit(X_train, y_train)
y_predict_lr = model_lin_reg.predict(X_test)
helpers.print_validation_metrics(y_test, y_predict_lr)


#%%
# =============================================================================
# TREINA E AVALIA MODELO COM DECISION TREE
# =============================================================================
# Melhores resultados com MinMaxScaler
model_dtr = DecisionTreeRegressor(
    max_depth=7, 
    criterion="absolute_error",
    splitter="best",
    min_samples_split=2,
    min_samples_leaf=3,
    random_state=71
)
model_dtr.fit(X_train_min_max_scaled, y_train)
y_predict_dtr = model_dtr.predict(X_test_min_max_scaled)
helpers.print_validation_metrics(y_test, y_predict_dtr)


#%%
# =============================================================================
# TREINA E AVALIA MODELO KNEIGHBORS REGRESSOR
# =============================================================================
# Melhores resultados com MinMaxScaler
model_knr = KNeighborsRegressor(
    n_neighbors=7,
    weights="uniform",
    leaf_size=40,
    p=2
)
model_knr.fit(X_train_min_max_scaled, y_train)
y_predict_knr = model_knr.predict(X_test_min_max_scaled)
helpers.print_validation_metrics(y_test, y_predict_knr)


#%%
# =============================================================================
# TREINA E AVALIA MODELO GAMMA REGRESSOR
# =============================================================================
# Melhores resultados com Standard scaler
model_gmr = GammaRegressor(
    alpha=0.09,
    solver="lbfgs"
)
model_gmr.fit(X_train_standard_scaled, y_train)
y_predict_gmr = model_gmr.predict(X_test_standard_scaled)
helpers.print_validation_metrics(y_test, y_predict_gmr)


#%%
# =============================================================================
# EXECUTA A VALIDAÇÃO CRUZADA DOS MODELOS
# =============================================================================
kfold  = KFold(n_splits=10, shuffle=True, random_state=71) 

lnr_result = cross_val_score(model_lin_reg, X, y, cv = kfold)
dtr_result = cross_val_score(model_dtr, X_min_max_scaled, y, cv = kfold)
knr_result = cross_val_score(model_knr, X_min_max_scaled, y, cv = kfold)
gmr_result = cross_val_score(model_gmr, X_standard_scaled, y, cv = kfold)

models_scores = {
    "LinearRegression": lnr_result.mean(),
    "DecisionTree": dtr_result.mean(),
    "KNRegression": knr_result.mean(),
    "GammaRegression": gmr_result.mean()
}

melhor_modelo = max(models_scores, key=models_scores.get)

print(f"O melhor modelo é : {melhor_modelo} com o valor: {models_scores[melhor_modelo]}")
# Decision Tree Regressor performou melhor, mas o KNRegressor não ficou tão ruim

#%%
# =============================================================================
# PLOTA VALORES REAIS VS PREDITOS DO MELHOR MODELO, DECISION TREE REGRESSION
# =============================================================================

plt.scatter(y_test, y_predict_dtr)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges')
plt.show()



residuals = y_test - y_predict_dtr
plt.scatter(y_predict_dtr, residuals)
plt.xlabel('Predicted Charges')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
