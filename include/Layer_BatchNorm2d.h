//
// Created by Koer on 2022/10/31.
//

#ifndef CRN_LAYER_BATCHNORM2D_H
#define CRN_LAYER_BATCHNORM2D_H


#include "Eigen"
#include "mat.h"
#include "Eigen/CXX11/Tensor"

class Layer_BatchNorm2d {
public:
    Layer_BatchNorm2d();

    Layer_BatchNorm2d(int64_t bn_ch);

    void LoadState(MATFile *pmFile, const std::string &state_preffix);

    void LoadTestState();

    Eigen::Tensor<float_t, 4> forward(Eigen::Tensor<float_t, 4> &input);

private:
    int64_t channels;
    Eigen::Tensor<float_t, 2> weights;
    Eigen::Tensor<float_t, 2> bias;
    Eigen::Tensor<float_t, 2> running_mean;
    Eigen::Tensor<float_t, 2> running_var;
    int32_t num_batches_tracked;


};


#endif //CRN_LAYER_BATCHNORM2D_H
