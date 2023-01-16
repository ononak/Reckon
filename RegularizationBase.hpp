//
//  RegularizationBase.h
//  ToolRegu
//
//  Created by Önder Nazım Onak on 24.01.2022.
//

#ifndef RegularizationBase_h
#define RegularizationBase_h

#include "LinAlgebraDef.hpp"
#include <tuple>

namespace regu {

class Regularization {

public:
  /*
    returns
   Vec x -> regularized solution
   double constraint_norm -> ||Lx||q
   double residual_norm -> ||Ax-y||p
   **/
  virtual std::tuple<Vec, double, double> solve(Vec y, Mat A, Mat L,
                                                double lambda) const = 0;
};

};     // namespace regu
#endif /* RegularizationBase_h */
