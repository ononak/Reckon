# ReguTool (Lp-Lq regularization)

Regularization is a method to adjust how closely a model fit to data. Adding an additional term that penalizes the loss function is one of the way of adjusting output model.
Linear regularization problems can be defined as follows

(**)  min_x {|y-Ax|_p + |lambda*Lx|_q}
  
Here p and q represent the norm of the vectors. x is the unknown source (or the model parameters) to be estimated, A is the forward model matrix with high condition number or it is rank deficient. L is the regularization operator which is used adjust the properties of the output model. lamda is a regularization parameter employed to determine the weight of the regularization term in the solution. y is the noisy measurements that can be expressed as follows;

y = Ax + n

where n is the measurement noise vector.

- If p = q = 2, then the problem (**) is called Tikhonov regularization or L2 regularization.

- If p = 2, q = 1, then the problem (**) is called L1 regularization, which promotes sparsity in the solution.

It is also possible to use 0 < p, q <=2 to estimate the unknown x. 

Solver for min_x {|y-Ax|_p + lambda*|Lx|_q}

 Reference: A GENERALIZED KRYLOV SUBSPACE METHOD FOR Lp-Lq MINIMIZATION
 A. LANZA, S. MORIGI, L. REICHEL, AND F. SGALLARI
 DOI:10.1137/140967982
 
 # Non-sparse model and L2 estimation
 
![noisy1](https://user-images.githubusercontent.com/17129016/212777392-6f88272f-680d-40c7-be17-8d8a07e4fc07.png)  

![sol1](https://user-images.githubusercontent.com/17129016/212777680-580933fe-2248-4460-9c1f-7c1c6c05906e.png)

# Sparse model & L1 estimation

![measure1](https://user-images.githubusercontent.com/17129016/212778179-8b13d6c3-9a78-4d8a-a5c4-b9094d80eebb.png)

![sparseModel](https://user-images.githubusercontent.com/17129016/212778201-7951cfdc-9d97-4547-a8b7-6aceea96d623.png)

![l1Sol1](https://user-images.githubusercontent.com/17129016/212778219-8c61a6c3-f3f6-40e3-ac39-f26dc4cb1e56.png)




