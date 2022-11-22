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

    NonLinearity(std::string type, bool inplace = false);

    NonLinearity(std::string type, int64_t val, bool inplace = false);

    NonLinearity(std::string type, float_t val, bool inplace = false);

    NonLinearity(std::string type, float_t beta, float_t threshold);

    void ELU(Eigen::Tensor<float_t, 4> &input, Eigen::Tensor<float_t, 4> &output);

    void ReLU(Eigen::Tensor<float_t, 4> &input, Eigen::Tensor<float_t, 4> &output);

    void PReLU(Eigen::Tensor<float_t, 4> &input, Eigen::Tensor<float_t, 4> &output);

    void Softplus(Eigen::Tensor<float_t, 4> &input, Eigen::Tensor<float_t, 4> &output);

    void Sigmoid(Eigen::Tensor<float_t, 4> &input, Eigen::Tensor<float_t, 4> &output);


private:
    std::string type;
    float_t alpha;
    int64_t channels;
    float_t beta;
    float_t threshold;
    bool inplace;

};


#endif //CRN_NONLINEARITY_H
