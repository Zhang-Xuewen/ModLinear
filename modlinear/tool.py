"""
Name: tool.py
Author: Xuewen Zhang
Date:at 24/04/2024
version: 1.0.0
Description: model linearization tool
"""

import casadi as cs
from .utils import continuous_to_discrete, jacobianest, getCasadiFunc

def cas_linearize(fun, x_dim, u_dim, c2d=False, ts=None, M=100):
    """
        Obtain the linearized A, B matrices for the continuous differentiation function.
    Args:
        fun (function): the continuous ode differential function with input x and u
        x_dim, u_dim (vector)(dim, 1): expand set-point.
        ts (float): sampling period of the discrete model
        M (int): RK4 in one sample time ts, do rk4 (h = ts/M) times, to achieve accurate results, M should be larger
    Notes: 
        if the fun is discrete ode already, then will give the A, B for the linearized discrete model
            c2d=False, ts=None, M=1
        elif the fun is continuous ode, and want to obtain continuous A, B
            c2d=False, ts=None, M=1
        elif the fun is continuous ode, and want to obtain discrete A, B
            c2d=True, specify the sampling period ts and M
    Usage:
        A_fcn, B_fcn are symbolic function, need to give expand point x, u to obtain A, B
        e.g.:  A = A_fcn(xs, us)
               B = B_fcn(xs, us)
    """ 
    x = cs.SX.sym('x', x_dim)
    u = cs.SX.sym('u', u_dim)
    
    if c2d:
        # give the symbolic discrete model and only if the fun is continuous
        model = getCasadiFunc(fun, [x_dim, u_dim], ['x', 'u'], 'model', rk4=True, Delta=ts, M=M)
    else:
        # give the symbolic discrete or continuous model depends on the fun is continuous or discrete
        model = getCasadiFunc(fun, [x_dim, u_dim], ['x', 'u'], 'model')
    
    # obtain the jacobian of x and u
    x_jacobi = cs.jacobian(model(x, u), x)
    u_jacobi = cs.jacobian(model(x, u), u)
    
    # construct the symbolic function so that can obtain the A, B at any given expand point
    A_fcn = cs.Function('A', [x, u], [x_jacobi])
    B_fcn = cs.Function('B', [x, u], [u_jacobi])
    
    return A_fcn, B_fcn
    

def linearize_continuous(fun, xs):
    """
        Obtain the linearized A, B matrices for the continuous differentiation function.
    Args:
        fun (function): the continuous ode differential function
        xs (vector)(dim, 1): expand set-point, xs should contain all states including state, inputs, etc.
        ts (float): sampling period of the discrete model
    """
    Jacobi, _ = jacobianest(fun, xs)
    
    nx = Jacobi.shape[0]
    
    A = Jacobi[:, :nx]
    B = Jacobi[:, nx:]
    return A, B


def linearize_c2d(fun, xs, C=None, D=None, ts=1):
    """
        Linearize the model and transform the continuous model to discrete model
    Args:
        fun (function): the continuous ode differential function
        xs (vector)(dim, 1): expand set-point, xs should contain all states including state, inputs, etc.
        C, D (matrix): the matrix of controlled output,  yk = C xk + D uk
        ts (float): sampling period of the discrete model
    """
    Jacobi, _ = jacobianest(fun, xs)
    
    nx = Jacobi.shape[0]
    
    A = Jacobi[:, :nx]
    B = Jacobi[:, nx:]
    C = C
    D = D
    
    A_dis, B_dis, C_dis, D_dis = continuous_to_discrete(A, B, C=C, D=D, ts=ts)
    return A_dis, B_dis, C_dis, D_dis
    
    
    
    