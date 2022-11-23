//
// Created by Koer on 2022/10/31.
//

#ifndef CRN_LAYER_CONV2D_H
#define CRN_LAYER_CONV2D_H


#include "vector"
#include "mat.h"
#include "Eigen"
#include "tuple"

#include "Eigen/CXX11/Tensor"

class Layer_Conv2d {
public:
    Layer_Conv2d();

    Layer_Conv2d(int64_t in_ch, int64_t out_ch, std::pair<int64_t, int64_t> kernel = std::make_pair(1, 1),
                 std::pair<int64_t, int64_t> stride = std::make_pair(1, 1),
                 std::pair<int64_t, int64_t> dilation = std::make_pair(1, 1),
                 std::pair<int64_t, int64_t> padding = std::make_pair(0, 0));

    void LoadState(MATFile *pmFile, const std::string &state_preffix);

    void LoadTestState();

    Eigen::Tensor<float_t, 4> forward(Eigen::Tensor<float_t, 4> &input);

private:
    int64_t in_channels;
    int64_t out_channels;
    std::pair<int64_t, int64_t> kernel_size;
    std::pair<int64_t, int64_t> stride;
    std::pair<int64_t, int64_t> dilation;
    std::pair<int64_t, int64_t> padding;
    Eigen::Tensor<float_t, 4> weights;
    Eigen::Tensor<float_t, 2> bias;

};


#endif //CRN_LAYER_CONV2D_H
