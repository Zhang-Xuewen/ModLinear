"""
Name: __init__.py
Author: Xuewen Zhang
Date:at 24/04/2024
version: 1.0.0
Description: Import the required modules for easy use.
required packages: control, casadi
"""

# Version for modlinear.
__version__ = "1.0.0"

# Add modules and some specific functions.
from .utils import continuous_to_discrete, jacobianest, getCasadiFunc, plot_matrix
from .tool import linearize_continuous, linearize_c2d, cas_linearize
from control import ss, c2d

