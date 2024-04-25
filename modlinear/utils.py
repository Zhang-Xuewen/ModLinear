"""
Name: utils.py
Author: Xuewen Zhang
Date:at 24/04/2024
version: 1.0.0
Description: Utils for model linearization
"""

import numpy as np
import casadi as cs
import warnings
import matplotlib.pyplot as plt
from control import ss, c2d



def continuous_to_discrete(A, B, C=None, D=None, ts=1):
    """
        Transform the obtained linear continuous system into discrete linear systems
        The discrete linear model is :
            (xk1 - xss) = A_dis * (xk - xss) + B_dis * (uk - uss) + M_dis * (pk - pss)
            (xk1 - xss) = A_dis * (xk - xss) + [B_dis, M_dis] * [(uk - uss)]
                                                                [(zk - zss)]
                    yk = C_dis * xk
        Args:
            A, B, C, D: the matrix of continuous linear systems
            ts: sampling time of the discrete linear system
    """
    if D is None:
        D = np.zeros((C.shape[0], B.shape[1]))
    sys_continuous = ss(A, B, C, D)
    sys_discrete = c2d(sys_continuous, ts)
    
    A_dis = sys_discrete.A 
    B_dis = sys_discrete.B
    C_dis = sys_discrete.C
    D_dis = sys_discrete.D
    return A_dis, B_dis, C_dis, D_dis


def jacobianest(fun, x0):
    """
    Compute Jacobi and A, B, C of linear continuous system:
        (xk1 - xss) = A * (xk - xss) + B * (uk - uss) + M * (pk - pss)
        eq to:
        (xk1 - xss) = A * (xk - xss) + [B, M] * [(uk - uss)]
                                                [(zk - zss)]
                yk  = C * xk
    Args:
        fun (function): continuous ode differential function, analytical function to differentiate 
        x0 (data)(dim, 1):  expand set-point, vector location at which to differentiate fun
    Returns:
        jac, err
    """
    
    max_step = 100
    step_ratio = 2.0000001
    
    nx = x0.shape[0] 
    
    # Get the state at the set-point
    xf = fun(x0)
    xf = xf[:]
    nf = xf.shape[0]
    
    if nf == 0:
        jac = np.zeros((0, nx)) 
        err = jac
    
    relative_delta = max_step * step_ratio ** np.arange(0, -26, -1)  
    n_steps = len(relative_delta)
    
    # total number of derivatives we will need to take
    jac = np.zeros((nf, nx))
    err = jac
    
    for i in range(nx):
        x0_i = x0[i]
        
        if x0_i != 0:
            delta = x0_i * relative_delta
        else:
            delta = relative_delta
    
        # evaluate at each step, centered around x0_i difference to give a second order estimate
        fdel = np.zeros((nf, n_steps))
        
        for j in range(n_steps):
            xf1 = fun(swapelement(x0, i, x0_i + delta[j]))
            xf2 = fun(swapelement(x0, i, x0_i - delta[j]))
            
            fdif = xf1 - xf2
            fdel[:, j] = fdif[:, 0]

        # these are pure second order estimates of the first derivative, for each trial delta.
        derest = np.multiply(fdel, 0.5/delta)
        
        # The error term on these estimates has a second order component, but also some 4th and 6th order terms in it.
        # Use Romberg exrapolation to improve the estimates to 6th order, as well as to provide the error estimate.
        
        # loop here, as rombextrap coupled with the trimming will get complicated otherwise.
        for j in range(nf):
            der_romb,errest = rombextrap(step_ratio, derest[j, :], [2, 4])
        
            # Trim off 3 estimates at each end of the scale
            nest = len(der_romb)
            trim = np.concatenate((np.arange(1, 4), np.arange(nest-2, nest)))
            tags = np.argsort(der_romb)
            der_romb = np.delete(der_romb, trim)
            tags = np.delete(tags, trim)
            errest = errest[tags]
            
            # Pick the estimate with the lowest predicted error
            ind = np.argmin(errest)
            err[j, i] = errest[ind]
            jac[j, i] = der_romb[ind]
            
    return jac, err
    
def swapelement(x, idx, value):
    """swaps value as element index, into the vector x"""
    x = x.copy()
    x[idx] = value
    return x


def rombextrap(step_ratio, der_init, rombexpon):
    """
        do romberg extrapolation for each estimate
        
        StepRatio - Ratio decrease in step
        der_init - initial derivative estimates
        rombexpon - higher order terms to cancel using the romberg step
        
        return:
            der_romb - derivative estimates returned
            errest - error estimates
            amp - noise amplification factor due to the romberg step
    """
    srinv = 1 / step_ratio
    
    # do nothing if no romberg terms
    nexpon = len(rombexpon)
    rmat = np.ones((nexpon + 2, nexpon + 1))
    
    # two romberg terms
    rmat[1, 1:3] = srinv ** np.array(rombexpon)
    rmat[2, 1:3] = srinv ** (2 * np.array(rombexpon))
    rmat[3, 1:3] = srinv ** (3 * np.array(rombexpon))
    
    # qr factorization used for the extrapolation as well as the uncertainty estimates
    qromb, rromb = np.linalg.qr(rmat, mode='reduced')
    
    # the noise amplification is further amplified by the Romberg step.
    # amp = cond(rromb);
    
    # this does the extrapolation to a zero step size.
    ne = len(der_init)
    rhs = vec2mat(der_init, nexpon+2, ne - (nexpon + 2))
    rombcoefs = np.linalg.solve(rromb, np.dot(np.transpose(qromb), rhs))
    der_romb = rombcoefs[0,:]
    
    # Uncertainty estimate of derivative prediction
    s = np.sqrt(np.sum((rhs - np.dot(rmat, rombcoefs))**2, axis=0))
    cov1 = np.sum((np.linalg.solve(rromb, np.eye(nexpon+1)))**2, axis=1)  # 1 spare dof
    errest = np.dot(s, 12.7062047361747) * np.sqrt(cov1[0])
    return der_romb,errest

    
def vec2mat(x, n, m):
    """forms the matrix M, such that M[i,j] = x[i+j-1]"""
    i, j = np.meshgrid(np.arange(1, n+1), np.arange(0, m))
    ind = i + j
    ind = ind.T
    mat = x[ind]
    if n == 1:
        mat = mat.T
    return mat


def getCasadiFunc(f, varsizes=None, varnames=None, funcname=None, rk4=False,
                  Delta=1, M=1, scalar=None, casaditype=None, wraps=None,
                  numpy=None):
    """
    Takes a function handle and turns it into a Casadi function.

    f should be defined to take a specified number of arguments and return a
    scalar, list, or numpy array. varnames, if specified, gives names to each
    of the inputs, but this is not required.

    sizes should be a list of how many elements are in each one of the inputs.

    Alternatively, instead of specifying varsizes, varnames, and funcname,
    you can pass a casadi.Function as wraps to copy these values from the other
    function.

    The numpy argument determines whether arguments are passed with numpy
    array semantics or not. By default, numpy=True, which means symbolic
    variables are passed as numpy arrays of Casadi scalar symbolics. This means
    your function should be written to accept (and should also return) numpy
    arrays. If numpy=False, the arguments are passed as Casadi symbolic
    vectors, which have slightly different semantics. Note that 'scalar'
    is a deprecated synonym for numpy.

    To choose what type of Casadi symbolic variables to use, pass
    casaditype="SX" or casaditype="MX". The default value is "SX" if
    numpy=True, and "MX" if numpy=True.
    """
    # Decide if user specified wraps.
    if wraps is not None:
        if not isinstance(wraps, cs.Function):
            raise TypeError("wraps must be a casadi.Function!")
        if varsizes is None:
            varsizes = [wraps.size_in(i) for i in range(wraps.n_in())]
        if varnames is None:
            varnames = [wraps.name_in(i) for i in range(wraps.n_in())]
        if funcname is None:
            funcname = wraps.name()

    # Pass the buck to the sub function.
    if varsizes is None:
        raise ValueError("Must specify either varsizes or wraps!")
    if funcname is None:
        funcname = "f"
    if numpy is None and scalar is not None:
        numpy = scalar
        warnings.warn("Passing 'scalar' is deprecated. Replace with 'numpy'.")
    symbols = __getCasadiFunc(f, varsizes, varnames, funcname,
                              numpy=numpy, casaditype=casaditype,
                              allowmatrix=True)
    args = symbols["casadiargs"]
    fexpr = symbols["fexpr"]

    # Evaluate function and make a Casadi object.
    fcasadi = cs.Function(funcname, args, [fexpr], symbols["names"],
                              [funcname])

    # Wrap with rk4 if requested.
    if rk4:
        frk4 = rk4model(fcasadi, args[0], args[1:], Delta, M)
        fcasadi = cs.Function(funcname, args, [frk4], symbols["names"],
                                  [funcname])

    return fcasadi



def __getCasadiFunc(f, varsizes, varnames=None, funcname="f", numpy=None,
                    casaditype=None, allowmatrix=True):
    """
    Core logic for getCasadiFunc and its relatives.

    casaditype chooses what type of casadi variable to use, while numpy chooses
    to wrap the casadi symbols in a NumPy array before calling f. Both
    numpy and casaditype are None by default; the table below shows what values
    are used in the various cases.

                  +----------------------+-----------------------+
                  |       numpy is       |       numpy is        |
                  |         None         |       not None        |
    +-------------+----------------------+-----------------------+
    | casaditype  | casaditype = "SX"    | casaditype = ("SX" if |
    |  is None    | numpy = True         |   numpy else "MX")    |
    +------------------------------------+-----------------------+
    | casaditype  | numpy = (False if    | warning issued if     |
    | is not None |   casaditype == "MX" |   numpy == True and   |
    |             |   else True)         |   casaditype == "MX"  |
    +------------------------------------+-----------------------+

    Returns a dictionary with the following entries:

    - casadiargs: a list of the original casadi symbolic primitives

    - numpyargs: a list of the numpy analogs of the casadi symbols. Note that
                 this is None if numpy=False.

    - fargs: the list of arguments passed to f. This is numpyargs if numpyargs
             is not None; otherwise, it is casadiargs.

    - fexpr: the casadi expression resulting from evaluating f(*fargs).

    - XX: is either casadi.SX or casadi.MX depending on what type was used
          to create casadiargs.

    - names: a list of string names for each argument.

    - sizes: a list of one- or two-element lists giving the sizes.
    """
    # Check names.
    if varnames is None:
        varnames = ["x%d" % (i,) for i in range(len(varsizes))]
    else:
        varnames = [str(n) for n in varnames]
    if len(varsizes) != len(varnames):
        raise ValueError("varnames must be the same length as varsizes!")

    # Loop through varsizes in case some may be matrices.
    realvarsizes = []
    for s in varsizes:
        goodInput = True
        try:
            s = [int(s)]
        except TypeError:
            if allowmatrix:
                try:
                    s = list(s)
                    goodInput = len(s) <= 2
                except TypeError:
                    goodInput = False
            else:
                raise TypeError("Entries of varsizes must be integers!")
        if not goodInput:
            raise TypeError("Entries of varsizes must be integers or "
                            "two-element lists!")
        realvarsizes.append(s)

    # Decide which Casadi type to use and whether to wrap as a numpy array.
    # XX is either casadi.SX or casadi.MX.
    if numpy is None and casaditype is None:
        numpy = True
        casaditype = "SX"
    elif numpy is None:
        numpy = False if casaditype == "MX" else True
    elif casaditype is None:
        casaditype = "SX" if numpy else "MX"
    else:
        if numpy and (casaditype == "MX") and WARN_NUMPY_MX:
            warnings.warn("Using a numpy array of casadi MX is almost always "
                          "a bad idea. Consider refactoring to avoid.")
    XX = dict(SX=cs.SX, MX=cs.MX).get(casaditype, None)
    if XX is None:
        raise ValueError("casaditype must be either 'SX' or 'MX'!")

    # Now make the symbolic variables. Make numpy versions if requested.
    casadiargs = [XX.sym(name, *size)
                  for (name, size) in zip(varnames, realvarsizes)]
    if numpy:
        numpyargs = [__casadi_to_numpy(x) for x in casadiargs]
        fargs = numpyargs
    else:
        numpyargs = None
        fargs = casadiargs

    # Evaluate the function and return everything.
    fexpr = safevertcat(f(*fargs))
    return dict(fexpr=fexpr, casadiargs=casadiargs, numpyargs=numpyargs, XX=XX,
                names=varnames, sizes=realvarsizes)


def __casadi_to_numpy(x, matrix=False, scalar=False):
    """
    Converts casadi symbolic variable x to a numpy array of scalars.
    
    If matrix=False, the function will guess whether x is a vector and return
    the appropriate numpy type. To force a matrix, set matrix=True. To use
    a numpy scalar when x is scalar, use scalar=True.
    """
    shape = None
    if not matrix:
        if scalar and x.is_scalar():
            shape = ()
        elif x.is_vector():
            shape = (x.numel(),)
    if shape is None:
        shape = x.shape
    y = np.empty(shape, dtype=object)
    if y.ndim == 0:
        y[()] = x # Casadi uses different behavior for x[()].
    else:
        for i in np.ndindex(shape):
            y[i] = x[i]
    return y

    
def safevertcat(x):
    """
    Safer wrapper for Casadi's vertcat.

    the input x is expected to be an iterable containing multiple things that
    should be concatenated together. This is in contrast to Casadi 3.0's new
    version of vertcat that accepts a variable number of arguments. We retain
    this (old, Casadi 2.4) behavior because it makes it easier to check types.

    If a single SX or MX object is passed, then this doesn't do anything.
    Otherwise, if all elements are numpy ndarrays, then numpy's concatenate
    is called. If anything isn't an array, then casadi.vertcat is called.
    """
    symtypes = set(["SX", "MX"])
    xtype = getattr(x, "type_name", lambda: None)()
    if xtype in symtypes:
        val = x
    elif (not isinstance(x, np.ndarray) and
          all(isinstance(a, np.ndarray) for a in x)):
        val = np.concatenate(x)
    else:
        val = cs.vertcat(*x)
    return val

# Now give the actual functions.
def rk4model(f, x0, par, Delta=1, M=1):
    """
    Does M RK4 timesteps of function f with variables x0 and parameters par.

    The first argument of f must be var, followed by any number of parameters
    given in a list in order.

    Note that var and the output of f must add like numpy arrays.
    """
    h = Delta / M
    x = x0
    j = 0
    while j < M:  # For some reason, a for loop creates problems here.
        k1 = f(x, *par)
        k2 = f(x + k1 * h / 2, *par)
        k3 = f(x + k2 * h / 2, *par)
        k4 = f(x + k3 * h, *par)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
        j += 1
    return x


def plot_matrix(matrix, matrixname='matrix', savedir='',figsize=(5, 5), show=False, cmap='Greens'):
    """Plot the matrix"""
    fig = plt.figure(figsize=figsize)
    plt.imshow(matrix, cmap=cmap, interpolation='nearest')
    plt.colorbar()  # Show colorbar
    plt.suptitle(matrixname)   
    plt.tight_layout() 
    plt.xlim(1, matrix.shape[1])
    plt.ylim(matrix.shape[0], 1)
    plt.xticks([1, matrix.shape[1]])
    plt.yticks([matrix.shape[0], 1])
    fig.savefig(savedir + f"{matrixname}.pdf")
    if show:
        plt.show()
    else:
        plt.close()