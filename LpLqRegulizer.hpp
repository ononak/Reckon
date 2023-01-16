//
//  LpLqRegulizer.hpp
//  ToolRegu
//
//  Created by Önder Nazım Onak on 31.10.2021.
//

#ifndef LpLqRegulizer_h
#define LpLqRegulizer_h

#include "RegularizationBase.hpp"

namespace regu {

/**
 Solver for min_x {|y-Ax|_p + |Lx|_q}

 % Reference: A GENERALIZED KRYLOV SUBSPACE METHOD FOR Lp-Lq MINIMIZATION
 % A. LANZA, S. MORIGI, L. REICHEL, AND F. SGALLARI
 % DOI:10.1137/140967982
 */
class LpLqRegulizer : public Regularization {
private:
  double p, q;

public:
  LpLqRegulizer(double pval, double qval);
  /**
     solve  min_{x }(||Ax-y||p+ lambda^2||Lx||q) regularization problem
   */
  std::tuple<Vec, double, double> solve(Vec y, Mat A, Mat L,
                                        double lambda) const override;

private:
  /**
     solve  min_{x} (||Ax-y||2+ lambda^2||Lx||2) regularization problem
   */
  Vec solve_l2l2_regularization(Vec y, Mat A, Mat L, double lambda) const;
};

};     // namespace regu
#endif /* LpLqRegulizer_hpp */
