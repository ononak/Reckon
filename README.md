# Reckon

Reckon is a library containing various algorithms for scientific applications.  


## Regularization

Regularization is a method used to control how closely a model fits the data, especially in ill-posed or ill-conditioned problems. One common approach is to modify the loss function by adding a penalty term that constrains the solution. This leads to the linear regularization problem, which can be expressed as:

![equation](https://github.com/ononak/SciTool/assets/17129016/e6623a28-9465-428a-9ea9-3986f7902d15)

<!-- $\min_{\mathbf{x} \in \mathbb{R}^{n}}\{\frac{1}{p}\|\mathbf{y} - \mathbf{A}\mathbf{x}\|_{p}^{p} + \frac{\lambda^{2}}{q}\|\mathbf{L}\mathbf{x}\|_{q}^{q}\} \quad \quad 0 < p,q \leq 2 \quad \mathbf{A} \in \mathbb{R}^{m \times n} \quad \mathbf{y} \in \mathbb{R}^{m} \quad \mathbf{L} : \mathbb{R}^{m} \rightarrow \mathbb{R}^{s}$ -->
  
Here, $p$ and $q$ represent the norm of the vectors. $\mathbf{x}$ is the unknown source (or the model parameters) to be estimated, $\mathbf{A}$ is the forward model matrix with high condition number or it is rank deficient. $\mathbf{L}$ is the regularization operator, which is used to adjust the properties of the output model. $\lambda$ is a regularization parameter employed to determine the weight of the regularization term in the solution. $\mathbf{y}$ is the noisy measurements that can be expressed as follows;

$\mathbf{y}= \mathbf{A}\mathbf{x} + \mathbf{n}$

Where $\mathbf{n}$ is the measurement noise vector.

- If $p = q = 2$, then the problem is called Tikhonov regularization or $L_{2}$ regularization that usually yields smooth estimation.

- If $p = 2$, $q = 1$, then the problem is called $L_{1}$ regularization, which promotes sparsity in the solution.

## Kalman Filter

Kalman filtering is an algorithm that allows us to estimate the state of a system based on observations or measurements.
The state of a system at time k evolved from the prior state at time k-1, expressed in the following form;

$\mathbf{x}(k)= \mathbf{A}\mathbf{x}(k-1) + \mathbf{B}\mathbf{u}(k-1) + \mathbf{w}(k-1)$

And the accompanying observation is defined as;

$\mathbf{y}(k)= \mathbf{H}\mathbf{x}(k) + \mathbf{v}(k-1)$

where

$p(\mathbf{w}) \sim \mathcal{N}(0,\mathbf{Q})$

$p(\mathbf{v}) \sim \mathcal{N}(0, \mathbf{R})$

The cycle of discrete Kalman filter.

![Kalman](https://github.com/ononak/SciTool/assets/17129016/3728fe3d-11d6-434f-bd87-504838daff51)
