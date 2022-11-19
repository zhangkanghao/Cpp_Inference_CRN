//
// Created by 65181 on 2022/10/31.
//

#include "iostream"
#include "../include/Layer_LSTM.h"

Layer_LSTM::Layer_LSTM() {
    this->input_size = 64;
    this->hidden_size = 64;
    this->num_layers = 2;
    this->direction = 1;
}

Layer_LSTM::Layer_LSTM(int32_t inp_size, int32_t hid_size, int32_t num_layer, bool bidirectional) {
    this->input_size = inp_size;
    this->hidden_size = hid_size;
    this->num_layers = num_layer;
    this->direction = bidirectional ? 2 : 1;


}

void Layer_LSTM::LoadState(MATFile *pmFile, const std::string &state_preffix) {

    for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
        std::string weight_hh_name = state_preffix + "_weight_hh_l" + std::to_string(layer_idx);
        std::string bias_hh_name = state_preffix + "_bias_hh_l" + std::to_string(layer_idx);
        std::string weight_ih_name = state_preffix + "_weight_ih_l" + std::to_string(layer_idx);
        std::string bias_ih_name = state_preffix + "_bias_ih_l" + std::to_string(layer_idx);
        mxArray *pa = matGetVariable(pmFile, weight_hh_name.c_str());
        auto *values = (float_t *) mxGetData(pa);
        long long dim1 = mxGetM(pa);
        long long dim2 = mxGetN(pa);
        Eigen::Tensor<float_t, 2> tmp_weight_hh(dim1, dim2);
        int idx = 0;
        for (int i = 0; i < dim2; i++) {
            for (int j = 0; j < dim1; j++) {
                tmp_weight_hh(j, i) = values[idx++];
            }
        }
        // std::cout << tmp_weight_hh << std::endl;
        this->weight_hh.push_back(tmp_weight_hh);

        pa = matGetVariable(pmFile, bias_hh_name.c_str());
        values = (float_t *) mxGetData(pa);
        dim1 = mxGetM(pa);
        dim2 = mxGetN(pa);
        Eigen::Tensor<float_t, 2> tmp_bias_hh(dim1, dim2);
        idx = 0;
        for (int i = 0; i < dim2; i++) {
            for (int j = 0; j < dim1; j++) {
                tmp_bias_hh(j, i) = values[idx++];
            }
        }
        // std::cout << tmp_bias_hh << std::endl;
        this->bias_hh.push_back(tmp_bias_hh);

        pa = matGetVariable(pmFile, weight_ih_name.c_str());
        values = (float_t *) mxGetData(pa);
        dim1 = mxGetM(pa);
        dim2 = mxGetN(pa);
        Eigen::Tensor<float_t, 2> tmp_weight_ih(dim1, dim2);
        idx = 0;
        for (int i = 0; i < dim2; i++) {
            for (int j = 0; j < dim1; j++) {
                tmp_weight_ih(j, i) = values[idx++];
            }
        }
        // std::cout << tmp_weight_ih << std::endl;
        this->weight_ih.push_back(tmp_weight_ih);

        pa = matGetVariable(pmFile, bias_ih_name.c_str());
        values = (float_t *) mxGetData(pa);
        dim1 = mxGetM(pa);
        dim2 = mxGetN(pa);
        Eigen::Tensor<float_t, 2> tmp_bias_ih(dim1, dim2);
        idx = 0;
        for (int i = 0; i < dim2; i++) {
            for (int j = 0; j < dim1; j++) {
                tmp_bias_ih(j, i) = values[idx++];
            }
        }
        // std::cout << tmp_bias_ih << std::endl;
        this->bias_ih.push_back(tmp_bias_ih);

    }


}

Eigen::Tensor<float_t, 3> Layer_LSTM::forward(Eigen::Tensor<float_t, 3> &input) {
    const Eigen::Tensor<size_t, 3>::Dimensions &dim_inp = input.dimensions();
    int32_t batch = dim_inp[0], seq_len = dim_inp[1], feat_len = dim_inp[2];
    for (int idx_layer = 0; idx_layer < this->num_layers; idx_layer++) {

    }
    Eigen::Tensor<float_t, 3> a(1, 1, 1);
    return a;
}
