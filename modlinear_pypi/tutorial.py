"""
Name: tutorial.py
Author: Xuewen Zhang
Date:at 25/04/2024
version: 1.0.0
Description: A tutorial to show how to achieve model linearization 
using modlinear toolbox.
required packages: modlinear
"""

import numpy as np
# model linearization toolbox 
import modlinear as ml


def example_cas_linear():
    # system parameters
    x_dim = 10
    up_dim = 2
    ts = 1

    # construct symbolic A, B function for linear continuous model
    A_fcn, B_fcn = ml.cas_linearize(fun, x_dim, up_dim)

    # construct symbolic A, B function for linear discrete model
    A_dis_fcn, B_dis_fcn = ml.cas_linearize(fun, x_dim, up_dim, c2d=True, ts=ts, M=100)

    # obtain the A_dis, B_dis at certain expand point
    ps = np.loadtxt('tutorial/expand-point/ps.txt').reshape(-1, 1)
    us = np.loadtxt('tutorial/expand-point/us.txt').reshape(-1, 1)
    xs = np.loadtxt('tutorial/expand-point/xs.txt').reshape(-1, 1)
    ups = np.vstack((us, ps))

    A = A_fcn(xs, ups).full()
    B = B_fcn(xs, ups).full()
    
    A_dis = A_dis_fcn(xs, ups).full()
    B_dis = B_dis_fcn(xs, ups).full()
    
    # Plot the matrices
    ml.plot_matrix(A, matrixname='A_cas', savedir='tutorial/results/')
    ml.plot_matrix(B, matrixname='B_cas', figsize=(2, 5), cmap='OrRd', savedir='tutorial/results/')
    ml.plot_matrix(A_dis, matrixname='A_dis_cas', savedir='tutorial/results/')
    ml.plot_matrix(B_dis, matrixname='B_dis_cas', figsize=(2, 5), cmap='OrRd', savedir='tutorial/results/')
    
    
    
def example_linear_c2d():
    # expand point
    ps = np.loadtxt('tutorial/expand-point/ps.txt').reshape(-1, 1)
    us = np.loadtxt('tutorial/expand-point/us.txt').reshape(-1, 1)
    xs = np.loadtxt('tutorial/expand-point/xs.txt').reshape(-1, 1)
    state = np.vstack((xs, us, ps))
    
    # sampling period of discrete model 
    ts = 1
    
    # C matrix
    C = np.zeros((2, 10))
    C[0, 1] = 1
    C[1, 2] = 1
    
    # obtain linearized discrete model
    A_dis, B_dis, C_dis, D_dis = ml.linearize_c2d(wrap_fun, state, C=C, ts=ts)
    
    # save the matrices
    # np.savetxt('A_dis.txt', A_dis)
    # np.savetxt('B_dis.txt', B_dis)
    # np.savetxt('C_dis.txt', C_dis)
    # np.savetxt('D_dis.txt', D_dis)
    
    # Plot the matrices
    ml.plot_matrix(A_dis, matrixname='A_dis', savedir='tutorial/results/')
    ml.plot_matrix(B_dis, matrixname='B_dis', figsize=(2, 5), cmap='OrRd', savedir='tutorial/results/')

    print('-'*30, 'Finished', '-'*30)



def example_linear_con():
    # expand point
    ps = np.loadtxt('tutorial/expand-point/ps.txt').reshape(-1, 1)
    us = np.loadtxt('tutorial/expand-point/us.txt').reshape(-1, 1)
    xs = np.loadtxt('tutorial/expand-point/xs.txt').reshape(-1, 1)
    state = np.vstack((xs, us, ps))
    
    # obtain linearized discrete model
    A, B = ml.linearize_continuous(wrap_fun, state)
    
    # save the matrices
    # np.savetxt('A.txt', A)
    # np.savetxt('B.txt', B)
    
    # Plot the matrices
    ml.plot_matrix(A, matrixname='A', savedir='tutorial/results/')
    ml.plot_matrix(B, matrixname='B', figsize=(2, 5), cmap='OrRd', savedir='tutorial/results/')

    print('-'*30, 'Finished', '-'*30)
    



def fun(x, u):
    """
        ode function, use your own example
    """
    return np.array(x*u + 0.3*u)


def wrap_fun(state):
    """ Wrap the ode function with one input """
    x = state[:10]
    u = state[10:]
    return fun(x, u)



if __name__ == '__main__':
    # tutorial of nonlinear continuous model to linear discrete model with numerical calculation
    example_linear_c2d()
    
    # tutorial of nonlinear continuous model to linear continuous model with numerical calculation
    example_linear_con()
    
    # tutorial of nonlinear continuous model to linear discrete and continuous model with symbolic calculation
    example_cas_linear()