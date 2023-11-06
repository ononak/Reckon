# SciTool - Scientific Tool

SciTool is a library containing various algorithms for scientific applications.  

## Lp-Lq regularization

Regularization is a method to adjust how closely a model fits to data. Adding an additional term that penalizes the loss function is one of the ways of adjusting the output model.
Linear regularization problems can be defined as follows.

(**)  min_x {|y-Ax|_p + lambda^2*|Lx|_q}
  
Here, p and q represent the norm of the vectors. x is the unknown source (or the model parameters) to be estimated, A is the forward model matrix with high condition number or it is rank deficient. L is the regularization operator, which is used to adjust the properties of the output model. lambda is a regularization parameter employed to determine the weight of the regularization term in the solution. y is the noisy measurements that can be expressed as follows;

y = Ax + n

Where n is the measurement noise vector.

- If p = q = 2, then the problem (**) is called Tikhonov regularization or L2 regularization.

- If p = 2, q = 1, then the problem (**) is called L1 regularization, which promotes sparsity in the solution.

Using 0 < p, q <=2, it is also possible to estimate the unknown x. 

Solver for min_x {|y-Ax|_p + lambda^2*|Lx|_q}

 Reference: A GENERALIZED KRYLOV SUBSPACE METHOD FOR Lp-Lq MINIMIZATION
 A. LANZA, S. MORIGI, L. REICHEL, AND F. SGALLARI
 DOI:10.1137/140967982
 
 # Non-sparse model
 
 ![Untitled](https://github.com/ononak/ReguTool/assets/17129016/8b0f41c8-3ce6-4c34-9e08-e58c09831aab)

# Sparse model

![Untitled](https://github.com/ononak/ReguTool/assets/17129016/86cd290d-02ac-42b3-9df2-88ba937c93e0)




