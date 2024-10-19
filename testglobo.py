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
import numpy
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
pd.set_option('display.max_columns', None)
df = pd.read_excel('/mnt/d/documentos/globo/base_desafio_cartola.xlsx', sheet_name='in')
colunas = list(df.columns.str.split(","))[0]
df = df[df.columns[0]].str.split(",").apply(lambda x: pd.Series(x))
df.columns = colunas

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
df.loc[df['sexo'] == 'NA', 'sexo'] = numpy.nan
df.loc[df['uf'] == 'NA', 'uf'] = numpy.nan
df.loc[df['device'] == 'NA', 'device'] = numpy.nan
df.loc[df['cartola_status'] == 'NA', 'cartola_status'] = numpy.nan
# Na na variável 'cartola_status' não apresenta informação relevante e será retirada
df = df.loc[df['cartola_status'].notna(), :]

# Localizando valores nulos nas variáveis numéricas
df.loc[df['tempo_total'] == 'NA', 'tempo_total'] = numpy.nan

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

len(df.loc[:,'user']) == len(df.loc[:,'user'].unique())
df.describe()


# %% Regressão Logística
# Inverterei o protocolo de primeiro fazer a analise exploratória e depois o treino do modelo preditivo,
# pois utilizarei os coeficientes para definir o que será exposto nos gráficos.
# Utilizarei a regressão logística como preditivo simples
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
print(device_mapping)
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
for uf in sorted(cartola_free['uf'].unique().tolist())[1:]:
    cartola_free[uf] = (cartola_free['uf'] == uf).astype('int8')
cartola_free['device'] = cartola_free['device'].map(device_mapping).astype('int8')
cartola_free['device_pc_e_m'] = (cartola_free['device'] == 1).astype('int8')
cartola_free['device_m'] = (cartola_free['device'] == 0).astype('int8')
cartola_free.drop(columns = ['device'], inplace = True)
cartola_free['cartola_status'].value_counts(dropna = False)
if list(cartola_free['cartola_status'].cat.categories) != ['Cartola Free', 'Cartola Pro']:
    cartola_free['cartola_status'] = cartola_free['cartola_status'].cat.reorder_categories(
    ['Cartola Free', 'Cartola Pro'], 
    ordered=True)

# Definindo variáveis independentes (X) e dependente (y)
X = n_cartola.loc[:,'cartola_status' != n_cartola.columns]
y = n_cartola['cartola_status'].cat.codes  # Convertendo a variável categórica para códigos numéricos

# Dividindo o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Para obter a significância dos coeficientes, usamos statsmodels
X_const = sm.add_constant(X_train)  # Adicionar uma constante para o intercepto
modelo_stats = sm.Logit(y_train, X_const).fit()
# print(modelo_stats.summary())
# """
#                            Results: Logit
# =====================================================================
# Model:               Logit             Method:            MLE        
# Dependent Variable:  y                 Pseudo R-squared:  0.516      
# Date:                2024-10-18 22:33  AIC:               1057.1621  
# No. Observations:    1507              BIC:               1179.4733  
# Df Model:            22                Log-Likelihood:    -505.58    
# Df Residuals:        1484              LL-Null:           -1044.6    
# Converged:           1.0000            LLR p-value:       4.8214e-214
# No. Iterations:      13.0000           Scale:             1.0000     
# ---------------------------------------------------------------------
#                       Coef.  Std.Err.    z     P>|z|   [0.025  0.975]
# ---------------------------------------------------------------------
# const                -1.8545   0.1465 -12.6588 0.0000 -2.1416 -1.5674
# dias                  0.0423   0.0246   1.7218 0.0851 -0.0059  0.0905
# pviews                0.0099   0.0034   2.9324 0.0034  0.0033  0.0166
# visitas              -0.0264   0.0140  -1.8852 0.0594 -0.0539  0.0010
# tempo_total          -0.0000   0.0000  -0.5230 0.6010 -0.0001  0.0000
# futebol               0.0000   0.0000   0.0081 0.9936 -0.0001  0.0001
# futebol_intenacional  0.0006   0.0002   3.6676 0.0002  0.0003  0.0009
# futebol_olimpico      0.0000   0.0000   0.2880 0.7734 -0.0001  0.0001
# blog_cartola          0.0163   0.0016   9.9339 0.0000  0.0131  0.0196
# atletismo             0.0001   0.0001   1.1159 0.2645 -0.0000  0.0002
# ginastica            -0.0001   0.0001  -1.5544 0.1201 -0.0003  0.0000
# judo                 -0.0000   0.0001  -0.2488 0.8035 -0.0001  0.0001
# natacao              -0.0002   0.0001  -1.6084 0.1078 -0.0004  0.0000
# basquete              0.0001   0.0000   1.4906 0.1361 -0.0000  0.0001
# handebol              0.0000   0.0001   0.0487 0.9612 -0.0002  0.0002
# volei                -0.0006   0.0002  -2.6806 0.0073 -0.0010 -0.0002
# tenis                 0.0001   0.0001   0.8930 0.3718 -0.0001  0.0003
# canoagem              0.0000   0.0002   0.2785 0.7807 -0.0003  0.0003
# saltos_ornamentais   -0.0003   0.0002  -1.2203 0.2224 -0.0007  0.0002
# home                  0.0000   0.0000   0.7689 0.4419 -0.0000  0.0001
# home_olimpiadas      -0.0001   0.0001  -1.7366 0.0825 -0.0003  0.0000
# device_pc_e_m         0.9705   0.1996   4.8613 0.0000  0.5792  1.3618
# device_m              0.7533   0.2000   3.7673 0.0002  0.3614  1.1452
# =====================================================================
# """

# Filtrando as variáveis com P > 0.05 para deixar o modelo mais simples
summary_df = modelo_stats.summary2().tables[1]
p_values = summary_df[summary_df['P>|z|'] < 0.05]
p_values = p_values.copy()
p_values.loc[:,'multiplicador'] = p_values.loc[:,'Coef.'].map(numpy.exp)
p_values = p_values.loc[:,['multiplicador', 'P>|z|']]
print(p_values)
# >>> print(p_values)
#                       multiplicador         P>|z|
# const                      0.156533  1.000518e-36
# pviews                     1.009988  3.363690e-03
# futebol_intenacional       1.000613  2.448809e-04
# blog_cartola               1.016467  2.964677e-23
# volei                      0.999410  7.349506e-03
# device_pc_e_m              2.639331  1.166291e-06
# device_m                   2.124016  1.650119e-04
significant_features = p_values.index.tolist()

# Para obter a significância dos coeficientes, usamos statsmodels
X_const = sm.add_constant(X_train)  # Adicionar uma constante para o intercepto
modelo_stats = sm.Logit(y_train, X_const[significant_features]).fit()
print(modelo_stats.summary2())

# Ajustando o modelo de regressão logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = modelo_stats.predict(sm.add_constant(X_test))

# Avaliando o modelo
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Exibindo os coeficientes do modelo
coefficients = pd.DataFrame(model.coef_[0], X.columns, columns=['Coefficient'])
print(coefficients)
