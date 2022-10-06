import pandas as pd 
import numpy as np
import scipy.stats as sst

class LinearRegression():

  def __init__(self,x,y):
    self.__y = y.to_numpy().reshape(-1,1)
    self.__x = np.c_[np.ones(self.__y.shape[0]),x.to_numpy()]
    self.__n = len(self.__y)
    self.__k = self.__x.shape[1] -1
    self.__gl = self.__n - self.__k 

    self.__betas = None
    self.__errores_estandar = None
    self.__estadistico_t = None
    self.__p_values = None
    self.__intervalos = None
    self.__b_0 = None
    self.__lista_betas = None

  def ajuste(self):
    self.__betas = (np.linalg.inv(self.__x.T@self.__x)@self.__x.T@self.__y)
    self.__b_0 = self.__betas.ravel().tolist()[0]
    self.__lista_betas = self.__betas.ravel().tolist()
  
  def prediccion(self,x_prediccion):
    self.ajuste()
    x_prediccion = pd.DataFrame(x_prediccion).to_numpy().tolist()[0]
    return self.__b_0 + sum([valor*coeficiente for valor,coeficiente in zip(self.__lista_betas[1:],x_prediccion)])

  def elementos(self):
    self.ajuste()
    self.__residuales = self.__y - (self.__x @ self.__betas)
    self.__SEC = np.sum(np.square(self.__x @ self.__betas - np.mean(self.__y)))
    self.__SRC = np.sum(np.square(self.__residuales))
    self.__STC = np.sum(np.square(self.__y - np.mean(self.__y)))

    self.__r_cuadrado = 1 - (self.__SRC / self.__STC)
    self.__covarianzas = (self.__SRC/self.__gl) * (np.linalg.inv(self.__x.T @ self.__x))
    self.__varianzas = np.diag(self.__covarianzas)
    self.__errores_estandar = np.sqrt(self.__varianzas)
    
    self.__estadistico_t = [coefi/error for (coefi,error) in zip(self.__lista_betas,self.__errores_estandar)]
    self.__p_values = [sst.t.sf(np.abs(t), self.__n-1)*2 for t in self.__estadistico_t]  
    
    self.__nivel_t = sst.t.ppf(1 - 0.05/2, df=self.__n - self.__k - 1 ) 
    self.__intervalos = [sorted([round(coef + (errcoef * self.__nivel_t),4),round(coef - (errcoef * self.__nivel_t),4)]) for (coef,errcoef) in zip(self.__lista_betas,self.__errores_estandar)]
    return self.__intervalos

  def resumen(self):
    self.ajuste()
    self.elementos()
    resultados = pd.DataFrame()
    resultados['Betas'] = self.__lista_betas
    resultados['Error Estandar'] = self.__errores_estandar
    resultados['T'] = self.__estadistico_t
    resultados['P-Valor'] = [round(valor,4) for valor in self.__p_values] 
    resultados['Intervalo de Confianza'] = self.__intervalos

    adicional = pd.DataFrame()
    adicional.index = ['Información Adicional']
    adicional['SEC'] = [round(i,0) for i in [self.__SEC]]
    adicional['SRC'] = [round(i,0) for i in [self.__SRC]]
    adicional['STC'] = [round(i,0) for i in [self.__STC]]
    adicional['R2'] = [round(i,2) for i in [self.__r_cuadrado]]
 
    print(resultados)
    print()
    print(adicional.T)

  def breusch_pagan(self):
    self.elementos()
    y_ = self.__residuales @ self.__residuales.T
    x_ = self.__x  
    betas_ = (np.linalg.inv(x_.T@ x_) @ x_.T @ y_)
    residuales = y_ - (x_ @ betas_)
    SRC_ = np.sum(np.square(residuales))
    STC_ = np.sum(np.square(y_ - np.mean(y_)))

    r_cuadrado_ = 1 - (SRC_ / STC_)
    estadistico = r_cuadrado_
    p_valor_prueba = sst.chi2.pdf(estadistico , self.__k) 
    resultados_prueba = pd.DataFrame()
    resultados_prueba.index = ["Estadistico","P - Valor"]
    resultados_prueba['Prueba Breuch Pagan'] = [round(estadistico,5),round(p_valor_prueba,5)]
    print(resultados_prueba)

  def jarque_bera(self):
    u = self.__residuos
    A = (sum(u@u.T@u)/self.__n) / (sum(u@u.T)/self.__n)**(3/2)
    K = (sum(u@u.T@u@u)/self.__n) / (sum(u@u.T)/self.__n)**(2)
    JB = self.__n*(((A**2)/6)+(((K-3)**2)/24))
    return JB 