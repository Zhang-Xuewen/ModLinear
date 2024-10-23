# Model linearization function toolbox

A wrapped package to linearize the nonlinear continuous/discrete model. Including **numerical** and **symbolic** calculations.

If you have questions, remarks, technical issues etc. feel free to use the issues page of this repository. I am looking forward to your feedback and the discussion.

> Github project: [link](https://github.com/QiYuan-Zhang/ModLinear)
>
> PyPI: [link](https://pypi.org/project/modlinear/)
>
> Introduction: [link](https://Zhang-Xuewen.github.io/my-toolbox/2024/04/25/Developed-modlinear.html)
>
> Please be sure to explicitly **acknowledge** its use if you incorporate it into your work.

--- 
## I. How to use

This package operates within the Python framework.

### 1. Required packages

- Numpy
- Matplotlib
- Control
- CasADi &emsp; &emsp;     <-- 3 <= __version__ <= 4

### 2. Usage

- Download the [*modlinear*](https://github.com/QiYuan-Zhang/ModLinear) file and save it to your project directory.

- Or install using pip

```
    pip install modlinear
```
Then you can use the modlinear in your python project.

## II. modlinear toolbox organization
```
. 
└── modlinear 
    ├── cas_linearize 
    ├── linearize_continuous 
    ├── linearize_c2d
    ├── continuous_to_discrete
    └── plot_matrix
``` 
Detailed introduction of each function can be found using `help` in python.

### 1. cas_linearize

> Symbolic calculation

Obtain the linearized continuous/discrete A, B symbolic functions for the continuous/discrete ODE.

- Continuous/discrete A, B from continuous ODE
- Discrete A, B from discrete ODE

Due to **symbolic functions**, the A, B at **any expand state** can be easily obtained by giving the state values.

### 2. linearize_continuous

> Numerical calculation

Obtain the linearized continuous A, B matrices for the continuous ODE.

### 3. linearize_c2d

> Numerical calculation

Obtain the linearized discrete A, B matrices for the continuous ODE.

### 4. continuous_to_discrete

> Numerical calculation

Obtain the discrete model from the continuous model, utilizing `control` package.

### 5. plot_matrix

Plot a matrix.

## II. Linearization process

1. Indicate the **set-point** that will be expanded: $x_{ss}, u_{ss}, p_{ss}$.
2. Compute the Jacobian of the system and obtain $A$, $B$, $M$, and $C$ matrix of the **continuous linear system**.
    > $(x_{t+1} - x_{ss}) = A  (x_t - x_{ss}) + B  (u_{t} - u_{ss}) + M  (p_{t} - p_{ss})$
    >
    > $y_k  = C  x_k$
    >
    > which equals to: $(x_{k+1} - x_{ss}) = A (x_k - x_{ss}) + [B, M] [u_k - u_{ss}, z_k -z_{ss}]^T$
3. Transform the **continuous linear system** to **discrete linear system** and obtain $A_{dis}$, $B_{dis}$, $M_{dis}$, and $C_{dis}$.
    > $(x_{k+1} - x_{ss}) = A_{dis}  (x_k - x_{ss}) + B_{dis}  (u_{k} - u_{ss}) + M_{dis}  (p_{k} - p_{ss})$
    >
    > $y_k  = C_{dis}  x_k$

Note: This procedure is applicable to all systems.

## III. Tutorial 

There is a [tutorial](https://github.com/QiYuan-Zhang/ModLinear/blob/main/tutorial.py) example to illustrate how to use the *modlinear* to linearize nonlinear models.


## License


This project is developed by `Xuewen Zhang` (xuewen.zhang741@outlook.com).

The project is released under the APACHE license. See [LICENSE](https://github.com/QiYuan-Zhang/ModLinear/blob/main/LICENSE) for details.

Copyright 2024 Xuewen Zhang

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
```
    http://www.apache.org/licenses/LICENSE-2.0
```
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
