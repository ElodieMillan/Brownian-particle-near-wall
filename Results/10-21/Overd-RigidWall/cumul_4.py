# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:33:13 2020

@author: a.alexandre
"""


import numpy as np
#import matplotlib.pyplot as plt
from scipy.integrate import quad


def potentiel(x) :
    '''potentiel V(x)'''
    return 0

def boltzmann(x,fonc):
    '''fonc(x)* poids de Boltzmann'''
    return fonc(x)*np.exp(-potentiel(x))

def identite(x) :
    return 1

def D_transverse(x) :
    return 1



def D_long(x) :
    '''Coeff de diffusion longitudinal '''
    return (1-x**2)

def fonc_A(x) :
    return -int_J(x)*np.exp(potentiel(x))/D_transverse(x)

def f_0(x) :
    return -np.exp(-potentiel(x))*(int_R(x) -moyenne_R )

def fonc_R (x):
    '''fonc utilisée pour int_R'''
    return quad(f_0, -1, x)[0]*np.exp(potentiel(x))/D_transverse(x)


def fonc_J (x):
    '''fonc utilisée pour int_J'''
    return  np.exp(-potentiel(x))/Z*(D_long(x)- moyenne_D_long)


def int_J(x) :
    return quad(fonc_J, -1, x)[0]

def int_R (x):
    return quad(fonc_R, -1, x)[0]

def int_A(x) :
    return quad(fonc_A, -1, x)[0]


def f_1(x) :
    return np.exp(-potentiel(x))*(int_A(x) -moyenne_A )

def fonc_cumul_4(x) :
    '''calcule C_40 tel que C(t) = C_40 t - C_41 '''
    return int_R (x)*(D_long(x)- moyenne_D_long)

def fonc_cumul_corr(x):
    '''calcule C_41 tel que C(t) = C_40 t - C_41 '''
    return f_1(x) * (D_long(x)- moyenne_D_long)
    
    
Z = quad(boltzmann, -1, 1, args=(identite))[0]
moyenne_D_long = quad(boltzmann, -1, 1, args=(D_long))[0]/Z
moyenne_R = quad(boltzmann, -1, 1, args=(int_R))[0]/Z
moyenne_A = quad(boltzmann, -1, 1, args=(int_A))[0]/Z

cumul_40 = -quad(boltzmann,-1,1,args=(fonc_cumul_4))[0]
cumul_41 = quad(boltzmann,-1,1,args=(fonc_cumul_corr))[0]
print( cumul_40 )
print( cumul_41 )

