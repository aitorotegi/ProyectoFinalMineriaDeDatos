import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.stats import chi2_contingency

from sklearn import set_config

#set_config(transform_output="pandas")

def isNan(v):
    return v is None or str(v) == 'nan' or str(v).strip() == ''

class limpiar_espacios(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if X_df[col].dtype == 'object':
                es_string = X_df[col].apply(lambda x: isinstance(x, str))
                X_df.loc[es_string,col] = X_df.loc[es_string,col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
        return X_df

    def get_feature_names_out(self, input_features=None):
        return input_features

class minusculizar(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if X_df[col].dtype == 'object':
                es_string = X_df[col].apply(lambda x: isinstance(x, str))
                X_df.loc[es_string,col] = X_df.loc[es_string,col].str.lower()
        return X_df

    def get_feature_names_out(self, input_features=None):
        return input_features

class quitar_simbolos(BaseEstimator, TransformerMixin):    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df = X_df.replace(to_replace=r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s]', value='', regex=True)
        return X_df

    def get_feature_names_out(self, input_features=None):
        return input_features

class quitar_tildes(BaseEstimator, TransformerMixin): 
    def __init__(self):
        self.reemplazos = {
            'á':'a','à':'a','ä':'a','â':'a','ã':'a','å':'a',
            'é':'e','è':'e','ë':'e','ê':'e',
            'í':'i','ì':'i','ï':'i','î':'i',
            'ó':'o','ò':'o','ö':'o','ô':'o','õ':'o',
            'ú':'u','ù':'u','ü':'u','û':'u',
            'ñ':'n',
            'ç':'c',
            'ß':'ss'
        }
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df = X_df.replace(self.reemplazos, regex=True)
        return X_df

    def get_feature_names_out(self, input_features=None):
        return input_features

class agrupar_por_inclusion(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if X_df[col].dtype != 'object':
                continue
            v_limpios = {}
            for v in X_df[col]:
                v_str = str(v).strip()
                if v_str and v_str.lower() != 'nan':
                    v_limpios[v_str] = v

            v_ordenados = sorted(v_limpios.keys(), key=len)
            mapeo = {}
            procesados = set()
            originales = []

            for i,corto in enumerate(v_ordenados):
                if corto in procesados:
                    continue
                if len(corto.split()) < 2:
                    continue
                for j in range(i + 1, len(v_ordenados)):
                    largo = v_ordenados[j]
                    if largo not in procesados and corto in largo:
                        mapeo[largo] = corto
                        procesados.add(largo)
                        originales.append(largo)

            def aplicar_mapeo(val):
                val_str = str(val).strip()
                return mapeo.get(val_str, val) 

            X_df[col] = X_df[col].apply(aplicar_mapeo)
        return X_df

    def get_feature_names_out(self, input_features=None):
        return input_features

class rellenar_secuencial(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if X_df[col].dtype == 'object':
                X_df[col] = X_df[col].ffill()
        return X_df

    def get_feature_names_out(self, input_features=None):
        return input_features

class rellenar_con_moda(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.valores_nulos = ["nan", "NaN", "null", "None", "", " ", None]
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        X_df.replace(self.valores_nulos, np.nan, inplace=True)
        self.modes_ = X_df.mode(axis=0).iloc[0]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df.replace(self.valores_nulos, np.nan, inplace=True)
        for col in X_df.columns:            
            X_df[col] = X_df[col].fillna(self.modes_[col])
        return X_df

    def get_feature_names_out(self, input_features=None):
        return input_features

class rellenar_con_constante(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.valores_nulos = ["nan", "NaN", "null", "None", "", " ", None]
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df.replace(self.valores_nulos, np.nan, inplace=True)
        for col in X_df.columns:
            if X_df[col].dtype == 'object':
                X_df[col] = X_df[col].fillna("unknown")
            else:
                X_df[col] = X_df[col].fillna(0)
        return X_df

    def get_feature_names_out(self, input_features=None):
        return input_features

class rellenar_moda_por_grupos(BaseEstimator, TransformerMixin):
    def __init__(self, col_agrupacion, cols_a_imputar):
        self.valores_nulos = ["nan", "NaN", "null", "None", "", " ", None]
        self.col_agrupacion = col_agrupacion
        self.cols_a_imputar = cols_a_imputar
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        X_df.replace(self.valores_nulos, np.nan, inplace=True)
        self.modes_gen_ = X_df.mode(axis=0).iloc[0]
        self.modes_ = {}

        for col_agrup, col_input in zip(self.col_agrupacion,self.cols_a_imputar):
            def moda_segura(x):
                m = x.mode()
                return m.iloc[0] if not m.empty else np.nan
                
            self.modes_[col_input] = X_df.groupby(col_agrup)[col_input].apply(moda_segura)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df.replace(self.valores_nulos, np.nan, inplace=True)

        for col_agrup, col_input in zip(self.col_agrupacion,self.cols_a_imputar):            
            modas = self.modes_[col_input]
            modas_mapeadas = X_df[col_agrup].map(modas)
            X_df[col_input] = X_df[col_input].fillna(modas_mapeadas)
            X_df[col_input] = X_df[col_input].fillna(self.modes_gen_[col_input]) 
                         
        return X_df

    def get_feature_names_out(self, input_features=None):
        return input_features

def v_cramer(X1,X2):
    contingencia = pd.crosstab(X1, X2)
    chi2, _, _, _ = chi2_contingency(contingencia)
    n = X1.shape[0]
    q = min(contingencia.shape) - 1
    v = np.sqrt(chi2/(q*n))
    return v


class DataFrameWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        X_trans = self.transformer.transform(X)
        cols = self.transformer.get_feature_names_out()
        return pd.DataFrame(X_trans, columns=cols, index=X.index)

class TransformadorSalarioBinario(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        if 'salary_range' in X_df.columns:
            X_df['salary_range'] = X_df['salary_range'].notnull().astype(int)
        
        return X_df

class discretizador_salario(BaseEstimator, TransformerMixin):   
    def separar(self,string):
        
        if pd.isna(string):
            return "null"
            
        string_strip = str(string).strip()
            
        if "-" not in string_strip:
            try:
                salary = int(string_strip)
            except:
                return "null"
                
        else:
            try:
                salary_range = string_strip.split("-")
                
                salary = (float(salary_range[0]) + float(salary_range[1])) / 2
            except ValueError:
                return "null"

        if salary < 50000:
            return "low"
        if salary > 120000:
            return "high"
        return "medium"            
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        for col in X_df.columns: # Esto no deberia ser un problema porque solo se lo vamos a aplicar a 1 columna
            X_df.loc[:,col] = X_df.loc[:,col].apply(self.separar)
               
        return X_df
class contar_palabras(BaseEstimator, TransformerMixin):    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        X_df = X_df.apply(lambda col: col.str.split(" ").str.len())
               
        return X_df

class contar_letras(BaseEstimator, TransformerMixin):    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()

        X_df = X_df.apply(lambda col: col.str.len())
               
        return X_df


class TransformadorSalarioBinario(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        if 'salary_range' in X_df.columns:
            X_df['salary_range'] = X_df['salary_range'].notnull().astype(int)
        
        return X_df

class discretizador_salario(BaseEstimator, TransformerMixin):   
    def separar(self,string):
        
        if pd.isna(string):
            return "null"
            
        string_strip = str(string).strip()
            
        if "-" not in string_strip:
            try:
                salary = int(string_strip)
            except:
                return "null"
                
        else:
            try:
                salary_range = string_strip.split("-")
                
                salary = (float(salary_range[0]) + float(salary_range[1])) / 2
            except ValueError:
                return "null"

        if salary < 50000:
            return "low"
        if salary > 120000:
            return "high"
        return "medium"            
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        for col in X_df.columns: # Esto no deberia ser un problema porque solo se lo vamos a aplicar a 1 columna
            X_df.loc[:,col] = X_df.loc[:,col].apply(self.separar)
               
        return X_df

class BinaryEncoder:
    
    def __init__(self):
        self.mapeos = {}
        self.num_bits = {}

    def fit(self, X, columnas):
        for col in columnas:
            categorias = X[col].dropna().unique()
            mapa = {}
            
            for i, cat in enumerate(categorias):
                mapa[cat] = i
            
            self.mapeos[col] = mapa
            n_categorias = len(categorias)
            
            if n_categorias > 1:
                bits = math.ceil(math.log2(n_categorias))
            else:
                bits = 1
            
            self.num_bits[col] = bits
        return self

    def transform(self, X):
        
        X_transformado = X.copy()

        for col in self.mapeos:
            mapa = self.mapeos[col]
            bits = self.num_bits[col]
            valores_numericos = X_transformado[col].map(mapa)
            valores_numericos = valores_numericos.fillna(0)
            valores_numericos = valores_numericos.astype(int)
            
            for i in range(bits):
                nueva_col = []
                
                for x in valores_numericos:
                    valor = (x >> i) & 1
                    nueva_col.append(valor)
                X_transformado[col + "_bin_" + str(i)] = nueva_col
            X_transformado = X_transformado.drop(columns=[col])

        return X_transformado

    def fit_transform(self, X, columnas):
        self.fit(X, columnas)
        return self.transform(X)

class BinaryEncoderSklearn(BaseEstimator, TransformerMixin):
    def __init__(self, columnas):
        self.columnas = columnas
        self.mapeos = {}
        self.num_bits = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()

        for col in self.columnas:
            categorias = X_df[col].dropna().unique()

            mapa = {}
            for i, cat in enumerate(categorias):
                mapa[cat] = i

            self.mapeos[col] = mapa

            n_categorias = len(categorias)

            if n_categorias > 1:
                bits = math.ceil(math.log2(n_categorias))
            else:
                bits = 1

            self.num_bits[col] = bits

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()

        nuevas_columnas = []

        for col in self.columnas:
            mapa = self.mapeos[col]
            bits = self.num_bits[col]

            valores_numericos = X_df[col].map(mapa)
            valores_numericos = valores_numericos.fillna(0)
            valores_numericos = valores_numericos.astype(int)

            datos_bits = {}

            for i in range(bits):
                datos_bits[col + "_bin_" + str(i)] = valores_numericos.apply(
                    lambda x: (x >> i) & 1
                )

            nuevas_columnas.append(pd.DataFrame(datos_bits, index=X_df.index))

        X_df = X_df.drop(columns=self.columnas)

        if len(nuevas_columnas) > 0:
            X_df = pd.concat([X_df] + nuevas_columnas, axis=1)

        return X_df.copy()

class SelectorCorrelacion(BaseEstimator, TransformerMixin):
    def __init__(self, k=8):
        self.k = k
        self.columnas_seleccionadas_ = None
        self.correlaciones_ = None

    def fit(self, X, y):
        X_df = pd.DataFrame(X).copy()
        y_series = pd.Series(y).reset_index(drop=True)
        X_df = X_df.reset_index(drop=True)

        correlaciones = {}

        for col in X_df.columns:
            try:
                corr = X_df[col].corr(y_series)
                correlaciones[col] = abs(corr)
            except:
                correlaciones[col] = 0

        self.correlaciones_ = pd.Series(correlaciones).sort_values(ascending=False)

        self.columnas_seleccionadas_ = (
            self.correlaciones_
            .head(self.k)
            .index
            .tolist()
        )

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return X_df[self.columnas_seleccionadas_]
        