# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:24:38 2018

@author: Pendragon
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:37:07 2018

@author: Mimisbrunnr
"""

import pandas as pd
import numpy as np
import scipy.optimize as sco
import scipy.linalg as scl

df_close = pd.read_csv('/Users/cytus/Desktop/BLM2/Close.csv', index_col = 'DATE')
df_return = np.log(df_close).diff(1)
df_Rm = df_return['SUV']
df_Rm2 = df_return['SPX']
df_Rm.drop(['12/31/1992'], axis = 0, inplace = True)
df_Rm2.drop(['12/31/1992'], axis = 0, inplace = True)
df_return.drop(['SPX','SUV'], axis = 1, inplace = True)
df_return.drop(['12/31/1992'], axis = 0, inplace = True)
df_close.drop(['SPX','SUV'], axis = 1, inplace = True)
#df_weight = pd.read_csv('G:/Software-W/WD-python/BLM/Weight.csv')
num = df_return.columns.size

def BLM(data):
    tau = 0.03

    Eye = np.eye(num)
    #Pick = np.mat(Pick)
    #Q = [0.0003, 0.0005, 0.0004, 0.0001, -0.0002, 0.0003, 0.0001]
    #Q = np.mat(Q).T
    Omega = np.diag(Sigma)
    invOmega = np.diag(1/Omega)
    Omega = np.diag(Omega)
    Omega = np.mat(Omega)
    invOmega = np.mat(invOmega)
    #np.linalg.eigvals(Eye + invOmega * Sigma)
    #L = np.linalg.cholesky(Eye + invOmega * Sigma)
    
    M1 = scl.solve(Sigma + Omega, Eye, True)
    #Pi_hat = Pi + Sigma * (Sigma + Omega).I * (Q - Pi)
    Pi_hat = Pi + Sigma * M1 * (Q - Pi)
    #M2 = scl.lu_solve(Eye + invOmega * Sigma, Eye, 0)
    Sigma_hat = Sigma * (Eye + tau * (Eye + invOmega * Sigma).I)
    #Sigma_hat = Sigma * (Eye + tau * M2)
    #Sigma_hat2 = Sigma + ((tau * Sigma).I + (tau * Omega).I).I
    
    return [Pi_hat, Sigma_hat]
"""
def VarianceBLM(weight):
    Sigma = BLM(temp_df)[1]
    weight = np.mat(weight)
    return (weight * Sigma * weight.T * 252)[0, 0]

def Variance(weight):
    Sigma = np.cov(temp_df, rowvar = False)
    Sigma = np.mat(Sigma)
    weight = np.mat(weight)
    return (weight * Sigma * weight.T * 252)[0, 0]
"""

def Mean(weight):
    weight = np.mat(weight)
    return -(weight * Pi)[0, 0] + risk_adv * (weight * Sigma * weight.T)[0, 0] / 2
    #return (weight * Pi * 252)[0, 0]

def MeanBLM(weight):
    [BLM_Pi, BLM_Sigma] = BLM(temp_df)
    weight = np.mat(weight)
    return -(weight * BLM_Pi)[0, 0] + risk_adv * (weight * BLM_Sigma * weight.T)[0, 0] / 2
    #return (weight * BLM_Pi * 252)[0, 0]
"""
def Variance(weight):
    weight = np.mat(weight)
    return (weight * Sigma * weight.T * 252)[0, 0]

def VarianceBLM(weight):
    [temp_Pi, temp_Sigma] = BLM(temp_df)
    weight = np.mat(weight)
    return (weight * temp_Sigma * weight.T * 252)[0, 0]
"""

def MVOptimizer():
    bnds = tuple((0,1) for x in range(num))
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})#,
            #{'type': 'eq', 'fun': lambda x: Variance(x) - 0.04} )
    return sco.minimize(Mean, num * [1./num,], method = 'SLSQP', bounds = bnds, constraints = cons)['x']

def MVOptimizerBLM():
    bnds = tuple((0,1) for x in range(num))
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}),
            #{'type': 'eq', 'fun': lambda x: VarianceBLM(x) - 0.04} )
    return sco.minimize(MeanBLM, num * [1./num,], method = 'SLSQP', bounds = bnds, constraints = cons)['x']

day = 2520
window = 21 * 6
assets1 = [1000000]
assets2 = [1000000]
#assets3 = [1000000]
temp_df = df_return[day-2520:day]
Pi = np.mean(temp_df)
Pi = np.mat(Pi).T
Sigma = np.cov(temp_df, rowvar = False)
Sigma = np.mat(Sigma)
temp_Rm = df_Rm[day-2520:day]
Rm = np.mean(temp_Rm) * 252
risk_adv = 1#np.mean(temp_Rm)/np.var(temp_Rm)
Q = np.mat(temp_df[-window:].mean()).T

position1 = MVOptimizer() * assets1[-1]
df_position1 = pd.DataFrame(position1)
position1 = position1 / np.array(df_close[day:day+1])
df2_position1 = pd.DataFrame(position1)
position2 = MVOptimizerBLM() * assets2[-1]
df_position2 = pd.DataFrame(position2)
position2 = position2 / np.array(df_close[day:day+1])
df2_position2 = pd.DataFrame(position2)
#position3 = np.array([1./num,]) * assets3[-1] / np.array(df_close[day:day+1])

day = 2521
while day < 6508:
    asset1 = np.sum(position1 * np.array(df_close[day:day+1]))
    assets1.append(asset1)
    asset2 = np.sum(position2 * np.array(df_close[day:day+1]))
    assets2.append(asset2)
    #asset3 = np.sum(position3 * np.array(df_close[day:day+1]))
    #assets3.append(asset3)
    if (day-2520)%21 == 0:
        temp_df = df_return[day-2520:day]
        Pi = np.mean(temp_df)
        Pi = np.mat(Pi).T
        Sigma = np.cov(temp_df, rowvar = False)
        Sigma = np.mat(Sigma)
        temp_Rm = df_Rm[day-2520:day]
        Rm = np.mean(temp_Rm) * 252
        #risk_adv = np.mean(temp_Rm)/np.var(temp_Rm)
        Q = np.mat(temp_df[-window:].mean()).T
        position1 = MVOptimizer() * assets1[-1]
        df_position1 = df_position1.append(pd.DataFrame(position1))
        position1 = position1 / np.array(df_close[day:day+1])
        df2_position1 = df2_position1.append(pd.DataFrame(position1))
        position2 = MVOptimizerBLM() * assets2[-1]
        df_position2 = df_position2.append(pd.DataFrame(position2))
        position2 = position2 / np.array(df_close[day:day+1])
        df2_position2 = df2_position2.append(pd.DataFrame(position2))
        #position3 = np.array([1./num,]) * assets3[-1] / np.array(df_close[day:day+1])
    day += 1
    print([day - 2520, asset1, asset2])






