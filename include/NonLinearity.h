//
// Created by Koer on 2022/10/31.
//

#ifndef CRN_NONLINEARITY_H
#define CRN_NONLINEARITY_H

#include "Eigen"
#include "Eigen/CXX11/Tensor"

class NonLinearity {
public:
    NonLinearity();

    Eigen::Tensor<float_t, 4> ELU(Eigen::Tensor<float_t, 4> &input, float_t alpha = 1.0);

    Eigen::Tensor<float_t, 4> ReLU(Eigen::Tensor<float_t, 4> &input);

    Eigen::Tensor<float_t, 4> Softplus(Eigen::Tensor<float_t, 4> &input, float_t beta = 1.0, float_t threshold = 20.0);

};


#endif //CRN_NONLINEARITY_H
