//
// Created by 65181 on 2022/10/31.
//

#ifndef CRN_LAYER_LSTM_H
#define CRN_LAYER_LSTM_H

#include "Eigen"
#include "mat.h"
#include "Eigen/CXX11/Tensor"

class Layer_LSTM {
public:
    Layer_LSTM();

    Layer_LSTM(int32_t inp_size, int32_t hid_size, int32_t num_layer = 2, bool bidirectional = false);

    void LoadState(MATFile *pmFile, const std::string &state_preffix);

    Eigen::Tensor<float_t, 3> forward(Eigen::Tensor<float_t, 3> &input);

private:
    int32_t input_size;
    int32_t hidden_size;
    int32_t num_layers;
    int32_t direction;
    std::vector<Eigen::Tensor<float_t, 2>> weight_hh, weight_ih;
    std::vector<Eigen::Tensor<float_t, 2>> bias_hh, bias_ih;

};


#endif //CRN_LAYER_LSTM_H
