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

    Layer_LSTM(int64_t inp_size, int64_t hid_size, int64_t num_layer = 2, bool bidirectional = false);

    void LoadState(MATFile *pmFile, const std::string &state_preffix);

    void LoadTestState();

    Eigen::Tensor<float_t, 3> forward(Eigen::Tensor<float_t, 3> &input, std::vector<Eigen::Tensor<float_t, 2>> &h_t,
                                      std::vector<Eigen::Tensor<float_t, 2>> &c_t);

private:
    int64_t input_size;
    int64_t hidden_size;
    int64_t num_layers;
    int64_t direction;
    bool bidirectional;
    std::vector<Eigen::Tensor<float_t, 2>> weight_ih, weight_hh;
    std::vector<Eigen::Tensor<float_t, 2>> weight_ih_reverse, weight_hh_reverse;
    std::vector<Eigen::Tensor<float_t, 2>> bias_ih, bias_hh;
    std::vector<Eigen::Tensor<float_t, 2>> bias_ih_reverse, bias_hh_reverse;

    Eigen::Tensor<float_t, 2> _load_mat(MATFile *pmFile, const std::string &state_name);

    Eigen::Tensor<float_t, 3> _uni_lstm(Eigen::Tensor<float_t, 3> &input, std::vector<Eigen::Tensor<float_t, 2>> &h_t,
                                        std::vector<Eigen::Tensor<float_t, 2>> &c_t);

    Eigen::Tensor<float_t, 3> _bi_lstm(Eigen::Tensor<float_t, 3> &input, std::vector<Eigen::Tensor<float_t, 2>> &h_t,
                                       std::vector<Eigen::Tensor<float_t, 2>> &c_t);


};


#endif //CRN_LAYER_LSTM_H
