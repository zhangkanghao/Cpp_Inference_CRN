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

Layer_LSTM::Layer_LSTM(int64_t inp_size, int64_t hid_size, int64_t num_layer, bool bidirectional) {
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

Eigen::Tensor<float_t, 3> Layer_LSTM::forward(Eigen::Tensor<float_t, 3> &input,
                                              std::vector<Eigen::Tensor<float_t, 2>> &h_t,
                                              std::vector<Eigen::Tensor<float_t, 2>> &c_t) {
    Eigen::Tensor<size_t, 3>::Dimensions dim_inp = input.dimensions();
    Eigen::Tensor<float_t, 3> out_pointer = input;
    if (h_t.empty() || c_t.empty()) {
        for (int idx_layer = 0; idx_layer < this->num_layers; idx_layer++) {
            Eigen::Tensor<float, 2> ht_zeros(dim_inp[0], this->hidden_size);
            Eigen::Tensor<float, 2> ct_zeros(dim_inp[0], this->hidden_size);
            ht_zeros.setZero();
            ct_zeros.setZero();
            h_t.push_back(ht_zeros);
            c_t.push_back(ct_zeros);
        }
    }
    for (int idx_layer = 0; idx_layer < this->num_layers; idx_layer++) {
        Eigen::Tensor<size_t, 3>::Dimensions dim_cur = out_pointer.dimensions();
        int64_t N_BATCH = dim_cur[0], N_TIME = dim_cur[1], N_FREQ = dim_cur[2], N_HIDDEN = this->hidden_size;
        Eigen::Tensor<float_t, 2> cur_w_ih = this->weight_ih[idx_layer];
        Eigen::Tensor<float_t, 2> cur_w_hh = this->weight_hh[idx_layer];
        Eigen::Tensor<float_t, 2> cur_b_ih = this->bias_ih[idx_layer].broadcast(Eigen::array<int64_t, 2>{N_BATCH, 1});
        Eigen::Tensor<float_t, 2> cur_b_hh = this->bias_hh[idx_layer].broadcast(Eigen::array<int64_t, 2>{N_BATCH, 1});
        Eigen::Tensor<float, 2> &cur_ht = h_t[idx_layer];
        Eigen::Tensor<float, 2> &cur_ct = c_t[idx_layer];


        Eigen::Tensor<float_t, 3> output(N_BATCH, N_TIME, N_HIDDEN);
        Eigen::Tensor<float_t, 2> X_t(N_BATCH, N_FREQ);
        Eigen::Tensor<float_t, 2> gates;
        Eigen::Tensor<float_t, 2> i_t, f_t, g_t, o_t;
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 1)};
        Eigen::array<int64_t, 2> gate_patch = Eigen::array<int64_t, 2>{N_BATCH, N_HIDDEN};
        for (int t = 0; t < N_TIME; t++) {
            X_t = input.chip(t, 1);
            gates = X_t.contract(cur_w_ih, product_dims) + cur_b_ih + cur_ht.contract(cur_w_hh, product_dims) +
                    cur_b_hh;
            i_t = gates.slice(Eigen::array<int64_t, 2>{0, N_HIDDEN * 0}, gate_patch).sigmoid();
            f_t = gates.slice(Eigen::array<int64_t, 2>{0, N_HIDDEN * 1}, gate_patch).sigmoid();
            g_t = gates.slice(Eigen::array<int64_t, 2>{0, N_HIDDEN * 2}, gate_patch).tanh();
            o_t = gates.slice(Eigen::array<int64_t, 2>{0, N_HIDDEN * 3}, gate_patch).sigmoid();
            cur_ct = f_t * cur_ct + i_t * g_t;
            cur_ht = o_t * cur_ct.tanh();
            output.chip(t, 1) = cur_ht;
        }
        out_pointer = output;
    }
    return out_pointer;
}


