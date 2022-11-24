//
// Created by 65181 on 2022/10/31.
//

#include "iostream"
#include "../include/Layer_LSTM.h"

using namespace std;

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
    this->bidirectional = bidirectional;
    this->direction = bidirectional ? 2 : 1;
}

void Layer_LSTM::LoadState(MATFile *pmFile, const std::string &state_preffix) {
    for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
        std::string weight_ih_name = state_preffix + "_weight_ih_l" + std::to_string(layer_idx);
        std::string bias_ih_name = state_preffix + "_bias_ih_l" + std::to_string(layer_idx);
        std::string weight_hh_name = state_preffix + "_weight_hh_l" + std::to_string(layer_idx);
        std::string bias_hh_name = state_preffix + "_bias_hh_l" + std::to_string(layer_idx);

        this->weight_ih.push_back(_load_mat(pmFile, weight_ih_name));
        this->bias_ih.push_back(_load_mat(pmFile, bias_ih_name));
        this->weight_hh.push_back(_load_mat(pmFile, weight_hh_name));
        this->bias_hh.push_back(_load_mat(pmFile, bias_hh_name));

        if (this->bidirectional) {
            std::string w_ih_reverse = state_preffix + "_weight_ih_l" + std::to_string(layer_idx) + "_reverse";
            std::string b_ih_reverse = state_preffix + "_bias_ih_l" + std::to_string(layer_idx) + "_reverse";
            std::string w_hh_reverse = state_preffix + "_weight_hh_l" + std::to_string(layer_idx) + "_reverse";
            std::string b_hh_reverse = state_preffix + "_bias_hh_l" + std::to_string(layer_idx) + "_reverse";

            this->weight_ih_reverse.push_back(_load_mat(pmFile, w_ih_reverse));
            this->bias_ih_reverse.push_back(_load_mat(pmFile, b_ih_reverse));
            this->weight_hh_reverse.push_back(_load_mat(pmFile, w_hh_reverse));
            this->bias_hh_reverse.push_back(_load_mat(pmFile, b_hh_reverse));
        }
    }
}

Eigen::Tensor<float_t, 2> Layer_LSTM::_load_mat(MATFile *pmFile, const std::string &state_name) {
    mxArray *pa = matGetVariable(pmFile, state_name.c_str());
    auto *values = (float_t *) mxGetData(pa);
    long long dim1 = mxGetM(pa);
    long long dim2 = mxGetN(pa);
    Eigen::Tensor<float_t, 2> matrix(dim1, dim2);
    int idx = 0;
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim1; j++) {
            matrix(j, i) = values[idx++];
        }
    }
    return matrix;
}


void Layer_LSTM::LoadTestState() {
    for (int layer = 0; layer < this->num_layers; layer++) {
        int64_t _ih_DIM = layer == 0 ? this->input_size : this->hidden_size * this->direction;
        Eigen::Tensor<float_t, 2> state_w_ih(this->hidden_size * 4, _ih_DIM);
        Eigen::Tensor<float_t, 2> state_w_hh(this->hidden_size * 4, this->hidden_size);
        Eigen::Tensor<float_t, 2> state_b_ih(1, this->hidden_size * 4);
        Eigen::Tensor<float_t, 2> state_b_hh(1, this->hidden_size * 4);
        state_w_ih.setConstant(2);
        state_w_hh.setConstant(2);
        state_b_ih.setConstant(1.0);
        state_b_hh.setConstant(1.0);
        this->weight_ih.push_back(state_w_ih);
        this->weight_hh.push_back(state_w_hh);
        this->bias_ih.push_back(state_b_ih);
        this->bias_hh.push_back(state_b_hh);

//        Eigen::Tensor<float_t, 2> state_w_ih_reverse(this->hidden_size * 4, _ih_DIM);
//        Eigen::Tensor<float_t, 2> state_w_hh_reverse(this->hidden_size * 4, this->hidden_size);
//        Eigen::Tensor<float_t, 2> state_b_ih_reverse(1, this->hidden_size * 4);
//        Eigen::Tensor<float_t, 2> state_b_hh_reverse(1, this->hidden_size * 4);
//        state_w_ih_reverse.setConstant(layer + 1);
//        state_w_hh_reverse.setConstant(layer + 2);
//        state_b_ih_reverse.setConstant(0.5);
//        state_b_hh_reverse.setConstant(1.0);
//        this->weight_ih_reverse.push_back(state_w_ih_reverse);
//        this->weight_hh_reverse.push_back(state_w_hh_reverse);
//        this->bias_ih_reverse.push_back(state_b_ih_reverse);
//        this->bias_hh_reverse.push_back(state_b_hh_reverse);
    }
}


Eigen::Tensor<float_t, 3> Layer_LSTM::forward(Eigen::Tensor<float_t, 3> &input,
                                              std::vector<Eigen::Tensor<float_t, 2>> &h_t,
                                              std::vector<Eigen::Tensor<float_t, 2>> &c_t) {
    Eigen::Tensor<float_t, 3> output;
    if (this->bidirectional) {
        output = this->_bi_lstm(input, h_t, c_t);
    } else {
        output = this->_uni_lstm(input, h_t, c_t);
    }
    return output;
}

Eigen::Tensor<float_t, 3> Layer_LSTM::_uni_lstm(Eigen::Tensor<float_t, 3> &input,
                                                vector<Eigen::Tensor<float_t, 2>> &h_t,
                                                vector<Eigen::Tensor<float_t, 2>> &c_t) {
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
        // cout << "cur_w_ih" << endl << cur_w_ih << endl;
        // cout << "cur_w_hh" << endl << cur_w_hh << endl;
        // cout << "cur_b_ih" << endl << cur_b_ih << endl;
        // cout << "cur_b_hh" << endl << cur_b_hh << endl;
        // cout << "cur_ht" << endl << cur_ht << endl;
        // cout << "cur_ct" << endl << cur_ct << endl;

        Eigen::Tensor<float_t, 3> output(N_BATCH, N_TIME, N_HIDDEN);
        Eigen::Tensor<float_t, 2> X_t;
        Eigen::Tensor<float_t, 2> gates;
        Eigen::Tensor<float_t, 2> i_t, f_t, g_t, o_t;
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 1)};
        Eigen::array<int64_t, 2> gate_patch = Eigen::array<int64_t, 2>{N_BATCH, N_HIDDEN};
        for (int t = 0; t < N_TIME; t++) {
            X_t = out_pointer.chip(t, 1);
            // cout << "X_t" << endl << X_t << endl;
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

Eigen::Tensor<float_t, 3> Layer_LSTM::_bi_lstm(Eigen::Tensor<float_t, 3> &input,
                                               vector<Eigen::Tensor<float_t, 2>> &h_t,
                                               vector<Eigen::Tensor<float_t, 2>> &c_t) {
    Eigen::Tensor<size_t, 3>::Dimensions dim_inp = input.dimensions();
    Eigen::Tensor<float_t, 3> out_pointer = input;
    if (h_t.empty() || c_t.empty()) {
        for (int idx_layer = 0; idx_layer < this->num_layers * this->direction; idx_layer++) {
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
        Eigen::Tensor<float_t, 2> cur_w_ih_reverse = this->weight_ih_reverse[idx_layer];
        Eigen::Tensor<float_t, 2> cur_w_hh = this->weight_hh[idx_layer];
        Eigen::Tensor<float_t, 2> cur_w_hh_reverse = this->weight_hh_reverse[idx_layer];
        Eigen::Tensor<float_t, 2> cur_b_ih = this->bias_ih[idx_layer].broadcast(Eigen::array<int64_t, 2>{N_BATCH, 1});
        Eigen::Tensor<float_t, 2> cur_b_ih_reverse = this->bias_ih_reverse[idx_layer].broadcast(
                Eigen::array<int64_t, 2>{N_BATCH, 1});
        Eigen::Tensor<float_t, 2> cur_b_hh = this->bias_hh[idx_layer].broadcast(Eigen::array<int64_t, 2>{N_BATCH, 1});
        Eigen::Tensor<float_t, 2> cur_b_hh_reverse = this->bias_hh_reverse[idx_layer].broadcast(
                Eigen::array<int64_t, 2>{N_BATCH, 1});

        Eigen::Tensor<float, 2> &cur_ht = h_t[idx_layer * 2];
        Eigen::Tensor<float, 2> &cur_ht_reverse = h_t[idx_layer * 2 + 1];
        Eigen::Tensor<float, 2> &cur_ct = c_t[idx_layer * 2];
        Eigen::Tensor<float, 2> &cur_ct_reverse = c_t[idx_layer * 2 + 1];
        // cout << "cur_w_ih" << endl << cur_w_ih << endl;
        // cout << "cur_w_hh" << endl << cur_w_hh << endl;
        // cout << "cur_b_ih" << endl << cur_b_ih << endl;
        // cout << "cur_b_hh" << endl << cur_b_hh << endl;
        // cout << "cur_ht" << endl << cur_ht << endl;
        // cout << "cur_ct" << endl << cur_ct << endl;

        Eigen::Tensor<float_t, 3> output(N_BATCH, N_TIME, N_HIDDEN);
        Eigen::Tensor<float_t, 3> output_reverse(N_BATCH, N_TIME, N_HIDDEN);
        Eigen::Tensor<float_t, 2> X_t;
        Eigen::Tensor<float_t, 2> X_t_reverse;
        Eigen::Tensor<float_t, 2> gates;
        Eigen::Tensor<float_t, 2> gates_reverse;
        Eigen::Tensor<float_t, 2> i_t, f_t, g_t, o_t;
        Eigen::Tensor<float_t, 2> i_t_reverse, f_t_reverse, g_t_reverse, o_t_reverse;
        Eigen::Tensor<float_t, 2> cur_cat;
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 1)};
        Eigen::array<int64_t, 2> gate_patch = Eigen::array<int64_t, 2>{N_BATCH, N_HIDDEN};
        for (int t = 0; t < N_TIME; t++) {
            X_t = out_pointer.chip(t, 1);
            X_t_reverse = out_pointer.chip(N_TIME - t - 1, 1);
            // cout << "X_t" << endl << X_t << endl;
            // cout << "X_t_reverse" << endl << X_t_reverse << endl;
            gates = X_t.contract(cur_w_ih, product_dims) + cur_b_ih + cur_ht.contract(cur_w_hh, product_dims) +
                    cur_b_hh;
            gates_reverse = X_t_reverse.contract(cur_w_ih_reverse, product_dims) + cur_b_ih_reverse +
                            cur_ht_reverse.contract(cur_w_hh_reverse, product_dims) +
                            cur_b_hh_reverse;
            i_t = gates.slice(Eigen::array<int64_t, 2>{0, N_HIDDEN * 0}, gate_patch).sigmoid();
            f_t = gates.slice(Eigen::array<int64_t, 2>{0, N_HIDDEN * 1}, gate_patch).sigmoid();
            g_t = gates.slice(Eigen::array<int64_t, 2>{0, N_HIDDEN * 2}, gate_patch).tanh();
            o_t = gates.slice(Eigen::array<int64_t, 2>{0, N_HIDDEN * 3}, gate_patch).sigmoid();
            i_t_reverse = gates_reverse.slice(Eigen::array<int64_t, 2>{0, N_HIDDEN * 0}, gate_patch).sigmoid();
            f_t_reverse = gates_reverse.slice(Eigen::array<int64_t, 2>{0, N_HIDDEN * 1}, gate_patch).sigmoid();
            g_t_reverse = gates_reverse.slice(Eigen::array<int64_t, 2>{0, N_HIDDEN * 2}, gate_patch).tanh();
            o_t_reverse = gates_reverse.slice(Eigen::array<int64_t, 2>{0, N_HIDDEN * 3}, gate_patch).sigmoid();


            cur_ct = f_t * cur_ct + i_t * g_t;
            cur_ht = o_t * cur_ct.tanh();
            cur_ct_reverse = f_t_reverse * cur_ct_reverse + i_t_reverse * g_t_reverse;
            cur_ht_reverse = o_t_reverse * cur_ct_reverse.tanh();
            output.chip(t, 1) = cur_ht;
            output_reverse.chip(N_TIME - t - 1, 1) = cur_ht_reverse;
        }

        out_pointer = output.concatenate(output_reverse, 2);
    }
    return out_pointer;
}






