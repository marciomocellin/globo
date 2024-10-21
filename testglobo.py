'''
Objetivos: 
• Realizar uma análise exploratória dos dados com o objetivo de encontrar informações úteis para 
    identificação de perfis de usuários propensos a assinarem o Cartola. Utilize técnicas de estatística e 
    visualização de dados para apoiar suas conclusões. 
• Ajustar um modelo preditivo simples para identificar os perfis de usuários propensos a assinarem o 
    Cartola. Apresente os métodos de limpeza e processamento de dados para gerar a entrada do 
    modelo preditivo. Justifique a escolha das técnicas e modelo utilizados. 
• Elaborar uma apresentação com a análise, resultados e sugestões de melhorias para o projeto desenvolvido. 
• Disponibilizar o código fonte utilizado no exercício. Sua solução deve ser organizada, legível e limpa, 
    evitando soluções com over-engineering. Tente ser sucinto, mas preze pela qualidade do código. 
'''
# %% Lendo e corrigindo o dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import statsmodels.api as sm
pd.set_option('display.max_columns', None)
df = pd.read_excel('/mnt/d/documentos/globo/base_desafio_cartola.xlsx', sheet_name='in')
colunas = list(df.columns.str.split(","))[0]
df = df[df.columns[0]].str.split(",").apply(lambda x: pd.Series(x))
df.columns = colunas
del colunas
# >>> colunas
# ['user', 'sexo', 'uf', 'idade', 'dias', 'pviews', 'visitas', 'tempo_total', 'device', 'futebol', 'futebol_intenacional',
# 'futebol_olimpico', 'blog_cartola', 'atletismo', 'ginastica', 'judo', 'natacao', 'basquete', 'handebol', 'volei', 'tenis',
# 'canoagem', 'saltos_ornamentais', 'home', 'home_olimpiadas', 'cartola_status']

# %% Corrigindo os tipos de dados
# Variáveis numéricas
df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
df['dias'] = pd.to_numeric(df['dias'], errors='coerce')
df['pviews'] = pd.to_numeric(df['pviews'], errors='coerce')
df['visitas'] = pd.to_numeric(df['visitas'], errors='coerce')
df['tempo_total'] = pd.to_numeric(df['tempo_total'], errors='coerce')
df['futebol'] = pd.to_numeric(df['futebol'], errors='coerce')
df['futebol_intenacional'] = pd.to_numeric(df['futebol_intenacional'], errors='coerce')
df['futebol_olimpico'] = pd.to_numeric(df['futebol_olimpico'], errors='coerce')
df['blog_cartola'] = pd.to_numeric(df['blog_cartola'], errors='coerce')
df['atletismo'] = pd.to_numeric(df['atletismo'], errors='coerce')
df['ginastica'] = pd.to_numeric(df['ginastica'], errors='coerce')
df['judo'] = pd.to_numeric(df['judo'], errors='coerce')
df['natacao'] = pd.to_numeric(df['natacao'], errors='coerce')
df['basquete'] = pd.to_numeric(df['basquete'], errors='coerce')
df['handebol'] = pd.to_numeric(df['handebol'], errors='coerce')
df['volei'] = pd.to_numeric(df['volei'], errors='coerce')
df['tenis'] = pd.to_numeric(df['tenis'], errors='coerce')
df['canoagem'] = pd.to_numeric(df['canoagem'], errors='coerce')
df['saltos_ornamentais'] = pd.to_numeric(df['saltos_ornamentais'], errors='coerce')
df['home'] = pd.to_numeric(df['home'], errors='coerce')
df['home_olimpiadas'] = pd.to_numeric(df['home_olimpiadas'], errors='coerce')
# Variáveis categóricas
df['sexo'] = df['sexo'].astype('category')
df['uf'] = df['uf'].astype('category')
df['device'] = df['device'].astype('category')
df['cartola_status'] = df['cartola_status'].astype('category')
# Corrigindo encoding
df['cartola_status'] = df['cartola_status'].str.encode('latin1').str.decode('utf-8')
# Localizando valores nulos nas variáveis categóricas
df.loc[df['sexo'] == 'NA', 'sexo'] = np.nan
df.loc[df['uf'] == 'NA', 'uf'] = np.nan
df.loc[df['device'] == 'NA', 'device'] = np.nan
df.loc[df['cartola_status'] == 'NA', 'cartola_status'] = np.nan
# Na na variável 'cartola_status' não apresenta informação relevante e será retirada
df = df.loc[df['cartola_status'].notna(), :]

# Localizando valores nulos nas variáveis numéricas
df.loc[df['tempo_total'] == 'NA', 'tempo_total'] = np.nan

# %% Análise descritiva
df['cartola_status'].unique()
df.loc[df['cartola_status'] == 'Cartola Free'].describe(include='category')
df.loc[df['cartola_status'] == 'Não Cartola'].describe(include='all') # Não tem informação de sexo e uf
df.loc[df['cartola_status'] == 'Cartola Pro'].describe(include='category')
df.describe(include='category')
df.describe(include='object')
df.describe(include='all')
df['sexo'].value_counts()
df['uf'].value_counts()
df['cartola_status'].value_counts(dropna = False)
# Remoção de outliers
df = df.loc[((df['idade'] < 100) & (df['idade'] > 0)) | df['idade'].isna(), :]

len(df.loc[:,'user']) == len(df.loc[:,'user'].unique())
df.describe()


# %% Regressão Logística
# Inverterei o protocolo de primeiro fazer a analise exploratória e depois o treino do modelo preditivo,
# pois utilizarei a significância da variáveis para definir o que será exposto nos gráficos.
# Há dois caminhos para o usuário assinar o cartola Pro, da Cartola Free para cartola Pro e Não Cartola para cartola Pro.
# A variável dependente será 'cartola_status' e as variáveis independentes serão as demais.
# Separando as bases e balanceando as amostras

n_cartola = df.loc[df['cartola_status'] != 'Cartola Free', ['dias', 'pviews', 'visitas',
       'tempo_total', 'device', 'futebol', 'futebol_intenacional',
       'futebol_olimpico', 'blog_cartola', 'atletismo', 'ginastica', 'judo',
       'natacao', 'basquete', 'handebol', 'volei', 'tenis', 'canoagem',
       'saltos_ornamentais', 'home', 'home_olimpiadas', 'cartola_status']].dropna()
valores = n_cartola['cartola_status'].value_counts(dropna = False)
n_cartola = n_cartola.groupby(['cartola_status']).sample(n = min(list(valores)), random_state = 42)
n_cartola['cartola_status'] = n_cartola['cartola_status'].astype('category')
# Convertendo a variável categórica 'device' para códigos numéricos
n_cartola['device'] = n_cartola['device'].astype('category')
device_codes = n_cartola['device'].cat.codes
df_device_n_cartola = pd.DataFrame({
    'device': n_cartola['device'],
    'device_code': device_codes
}).drop_duplicates()
device_mapping = df_device_n_cartola.set_index('device').to_dict()['device_code']
n_cartola['device'] = device_codes
n_cartola['device_pc_e_m'] = (n_cartola['device'] == 1).astype('int8')
n_cartola['device_m'] = (n_cartola['device'] == 0).astype('int8')
n_cartola.drop(columns = ['device'], inplace = True)
n_cartola['cartola_status'].value_counts(dropna = False)
if list(n_cartola['cartola_status'].cat.categories) != ['Não Cartola', 'Cartola Pro']:
    n_cartola['cartola_status'] = n_cartola['cartola_status'].cat.reorder_categories(
        ['Não Cartola', 'Cartola Pro'], 
        ordered=True)

cartola_free = df.loc[df['cartola_status'] != 'Não Cartola', ['sexo', 'uf', 'idade', 'dias', 'pviews', 'visitas',
       'tempo_total', 'device', 'futebol', 'futebol_intenacional',
       'futebol_olimpico', 'blog_cartola', 'atletismo', 'ginastica', 'judo',
       'natacao', 'basquete', 'handebol', 'volei', 'tenis', 'canoagem',
       'saltos_ornamentais', 'home', 'home_olimpiadas', 'cartola_status']].dropna()
valores = cartola_free['cartola_status'].value_counts(dropna = False)
cartola_free = cartola_free.groupby(['cartola_status']).sample(n = min(list(valores)), random_state = 42)
cartola_free['cartola_status'] = cartola_free['cartola_status'].astype('category')
uf_mapping = cartola_free['uf'].unique().tolist()
uf_mapping = {uf: idx for idx, uf in enumerate(uf_mapping)}
cartola_free['uf'] = cartola_free['uf'].map(uf_mapping).astype('int8')
cartola_free['device'] = cartola_free['device'].map(device_mapping).astype('int8')
cartola_free['device_pc_e_m'] = (cartola_free['device'] == 1).astype('int8')
cartola_free['device_m'] = (cartola_free['device'] == 0).astype('int8')
sexo_mapping = {'F': 0, 'M': 1}
cartola_free['sexo'] = cartola_free['sexo'].map(sexo_mapping).astype('int8')
cartola_free.drop(columns = ['device',
                            'tempo_total',
                            'sexo'], inplace = True)
cartola_free['cartola_status'].value_counts(dropna = False)
if list(cartola_free['cartola_status'].cat.categories) != ['Cartola Free', 'Cartola Pro']:
    cartola_free['cartola_status'] = cartola_free['cartola_status'].cat.reorder_categories(
    ['Cartola Free', 'Cartola Pro'], 
    ordered=True)
# dataset = n_cartola.copy(deep=True)
def preditivo_simple(dataset: pd.DataFrame, var_interesse:str = 'cartola_status', save_pickle: bool = False):
    warnings.filterwarnings('ignore')
    # Definindo variáveis independentes (X) e dependente (y)
    X = dataset.loc[:,var_interesse != dataset.columns]
    y = dataset[var_interesse].cat.codes  # Convertendo a variável categórica para códigos numéricos
    # Dividindo o dataset em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Para obter a significância dos coeficientes, usamos statsmodels
    X_const = sm.add_constant(X_train)  # Adicionar uma constante para o intercepto
    modelo_stats = sm.Logit(y_train, X_const).fit()
    summary_df = modelo_stats.summary2().tables[1]
    p_values = summary_df[summary_df['P>|z|'] < 0.05]
    p_values['multiplicador'] = p_values.loc[:,'Coef.'].map(np.exp)
    p_values = p_values.loc[:,['multiplicador', 'P>|z|']]
    significant_features = p_values.index.tolist()
    # Para obter a significância dos coeficientes, usamos statsmodels
    X_const = sm.add_constant(X_train)[significant_features]  # Adicionar uma constante para o intercepto
    modelo_stats = sm.Logit(y_train, X_const).fit()
    if save_pickle:
        modelo_stats.save('modelo_stats.pickle')
    multiplicador = modelo_stats.summary2().tables[1]
    multiplicador['multiplicador'] = multiplicador.loc[:,'Coef.'].map(np.exp)
    # Fazendo previsões
    y_pred = modelo_stats.predict(sm.add_constant(X_test)[significant_features])
    # Convertendo para valores binários
    y_pred = (y_pred >= 0.5).astype(int)
    # Avaliando o modelo
    print(f"{multiplicador.loc[:,'multiplicador']}\n{classification_report(y_test, y_pred)}")

# %% Cartola Free
print(f"Cartola Free\n{preditivo_simple(cartola_free)}")

# %% Não Cartola
print(f"Não Cartola\n{preditivo_simple(n_cartola)}")

# 
# >>> print('Cartola Free')
# Cartola Free
# >>> preditivo_simple(cartola_free)
# Optimization terminated successfully.
#          Current function value: 0.634276
#          Iterations 7
# Optimization terminated successfully.
#          Current function value: 0.647480
#          Iterations 6
# const              0.251266
# idade              1.033344
# blog_cartola       1.000055
# home_olimpiadas    0.999997
# device_pc_e_m      2.287302
# Name: multiplicador, dtype: float64
#               precision    recall  f1-score   support
# 
#            0       0.58      0.58      0.58       243
#            1       0.56      0.56      0.56       231
# 
#     accuracy                           0.57       474
#    macro avg       0.57      0.57      0.57       474
# weighted avg       0.57      0.57      0.57       474
# 
# >>> print('Não Cartola')
# Não Cartola
# >>> preditivo_simple(n_cartola)
# Optimization terminated successfully.
#          Current function value: 0.335488
#          Iterations 13
# Optimization terminated successfully.
#          Current function value: 0.350842
#          Iterations 13
# const                   0.169647
# pviews                  1.003242
# futebol_intenacional    1.000277
# blog_cartola            1.016878
# volei                   0.999266
# device_pc_e_m           2.407832
# device_m                2.126595
# Name: multiplicador, dtype: float64
#               precision    recall  f1-score   support
# 
#            0       0.78      0.98      0.87       323
#            1       0.97      0.73      0.83       324
# 
#     accuracy                           0.85       647
#    macro avg       0.87      0.85      0.85       647
# weighted avg       0.87      0.85      0.85       647
 

# %% Análise exploratória
# Cartola Free (idade, blog_cartola, home_olimpiadas, device_pc_e_m)
# percentual de usuários que assinaram o Cartola Pro por idade
df_analise = cartola_free.loc[:, ['idade', 'cartola_status']]
df_analise['cartola_status'] = df_analise['cartola_status'].cat.codes
df_analise.groupby('idade').aggregate({'cartola_status': 'mean'}).plot()
plt.title('Percentual de usuários que assinaram o Cartola Pro por idade (Cartola Free)')
plt.show()

# %% # percentual de usuários que assinaram o Cartola Pro por blog_cartola

df_analise = cartola_free.loc[:, ['blog_cartola', 'cartola_status']]
df_analise['cartola_status'] = df_analise['cartola_status'].cat.codes
df_analise.groupby('blog_cartola').aggregate({'cartola_status': 'mean'}).plot()
plt.title('Percentual de usuários que assinaram o Cartola Pro por blog_cartola (Cartola Free)')
plt.show()
# %%
# cartola_status vs idade
fig, ax = plt.subplots()
sns.violinplot(x='cartola_status', y='idade', data=cartola_free, ax=ax)
ax.set_title('Idade vs Cartola Status (Cartola Free)')
plt.show()

# %% cartola_status vs blog_cartola
fig, ax = plt.subplots()
sns.violinplot(x='cartola_status', y='blog_cartola', data=cartola_free, ax=ax)
ax.set_title('Blog Cartola vs Cartola Status (Cartola Free)')
plt.show()

# %% cartola_status vs home_olimpiadas
fig, ax = plt.subplots()
sns.violinplot(x='cartola_status', y='home_olimpiadas', data=cartola_free, ax=ax)
ax.set_title('Home Olimpíadas vs Cartola Status (Cartola Free)')
plt.show()

# %% percentual de usuários que assinaram o Cartola Pro por device
df_analise = df.loc[df['cartola_status'] != 'Não Cartola', [ 'device','cartola_status']].dropna()
valores = df_analise['cartola_status'].value_counts(dropna = False)
df_analise = df_analise.groupby(['cartola_status']).sample(n = min(list(valores)), random_state = 42)
df_analise['cartola_status'] = df_analise['cartola_status'].astype('category')
if list(n_cartola['cartola_status'].cat.categories) != ['Não Cartola', 'Cartola Pro']:
    n_cartola['cartola_status'] = n_cartola['cartola_status'].cat.reorder_categories(
        ['Não Cartola', 'Cartola Pro'], 
        ordered=True)
df_analise['cartola_status'] = df_analise['cartola_status'].cat.codes
df_analise.groupby('device').aggregate({'cartola_status': 'mean'}).plot(kind='bar')
plt.title('Percentual de usuários que assinaram o Cartola Pro por device')
plt.show()

# Não Cartola (pviews, futebol_intenacional, blog_cartola, volei, device_pc_e_m, device_m)
# %% cartola_status vs pviews
fig, ax = plt.subplots()
sns.violinplot(x='cartola_status', y='pviews', data=n_cartola, ax=ax)
ax.set_title('Pviews vs Cartola Status (Não Cartola)')
plt.show()

# %% cartola_status vs futebol_intenacional
fig, ax = plt.subplots()
sns.violinplot(x='cartola_status', y='futebol_intenacional', data=n_cartola, ax=ax)
ax.set_title('Futebol Internacional vs Cartola Status (Não Cartola)')
plt.show()

# %% cartola_status vs blog_cartola
fig, ax = plt.subplots()
sns.violinplot(x='cartola_status', y='blog_cartola', data=n_cartola, ax=ax)
ax.set_title('Blog Cartola vs Cartola Status (Não Cartola)')
plt.show()

# %% cartola_status vs volei
fig, ax = plt.subplots()
sns.violinplot(x='cartola_status', y='volei', data=n_cartola, ax=ax)
ax.set_title('Volei vs Cartola Status (Não Cartola)')
plt.show()

# %% cartola_status vs device_pc_e_m
fig, ax = plt.subplots()
sns.violinplot(x='cartola_status', y='device_pc_e_m', data=n_cartola, ax=ax)
ax.set_title('Device PC e M vs Cartola Status (Não Cartola)')
plt.show()

# %% cartola_status vs device_m
fig, ax = plt.subplots()
sns.violinplot(x='cartola_status', y='device_m', data=n_cartola, ax=ax)
ax.set_title('Device M vs Cartola Status (Não Cartola)')
plt.show()

# %% percentual de usuários que assinaram o Cartola Pro por device
df_analise = df.loc[df['cartola_status'] != 'Cartola Free', [ 'device','cartola_status']].dropna()
valores = df_analise['cartola_status'].value_counts(dropna = False)
df_analise = df_analise.groupby(['cartola_status']).sample(n = min(list(valores)), random_state = 42)
df_analise['cartola_status'] = df_analise['cartola_status'].astype('category')
if list(df_analise['cartola_status'].cat.categories) != ['Não Cartola', 'Cartola Pro']:
    df_analise['cartola_status'] = df_analise['cartola_status'].cat.reorder_categories(
        ['Não Cartola', 'Cartola Pro'], 
        ordered=True)
df_analise['cartola_status'] = df_analise['cartola_status'].cat.codes
df_analise.groupby('device').aggregate({'cartola_status': 'mean'}).plot(kind='bar')
plt.title('Percentual de usuários que assinaram o Cartola Pro por device')
plt.show()


# %% percentual de usuários que assinaram o Cartola Pro por uf
df_analise = df.loc[df['cartola_status'] != 'Não Cartola', [ 'uf','cartola_status']].dropna()
valores = df_analise['cartola_status'].value_counts(dropna = False)
df_analise = df_analise.groupby(['cartola_status']).sample(n = min(list(valores)), random_state = 42)
df_analise['cartola_status'] = df_analise['cartola_status'].astype('category')
if list(cartola_free['cartola_status'].cat.categories) != ['Cartola Free', 'Cartola Pro']:
    cartola_free['cartola_status'] = cartola_free['cartola_status'].cat.reorder_categories(
    ['Cartola Free', 'Cartola Pro'], 
    ordered=True)
df_analise['cartola_status'] = df_analise['cartola_status'].cat.codes
df_analise.groupby('uf').aggregate({'cartola_status': 'mean'}).plot(kind='bar')
plt.title('Percentual de usuários que assinaram o Cartola Pro por uf')
plt.show()

# %%
