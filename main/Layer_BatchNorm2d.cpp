//
// Created by Koer on 2022/10/31.
//
#include "iostream"
#include "../include/Layer_BatchNorm2d.h"

Layer_BatchNorm2d::Layer_BatchNorm2d() {
    this->channels = 1;
}

Layer_BatchNorm2d::Layer_BatchNorm2d(int64_t bn_ch) {
    this->channels = bn_ch;
}

void Layer_BatchNorm2d::LoadState(MATFile *pmFile, const std::string &state_preffix) {
    std::string weight_name = state_preffix + "_weight";
    std::string bias_name = state_preffix + "_bias";
    std::string rm_name = state_preffix + "_running_mean";
    std::string rv_name = state_preffix + "_running_var";
    std::string nbt_name = state_preffix + "_num_batches_tracked";

    mxArray *pa = matGetVariable(pmFile, weight_name.c_str());
    auto *values = (float_t *) mxGetData(pa);
    long long dim1 = mxGetM(pa);
    long long dim2 = mxGetN(pa);
    this->weights.resize(dim1, dim2);
    int idx = 0;
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim1; j++) {
            this->weights(j, i) = values[idx++];
        }
    }
    // std::cout << this->weights << std::endl;

    pa = matGetVariable(pmFile, bias_name.c_str());
    values = (float_t *) mxGetData(pa);
    dim1 = mxGetM(pa);
    dim2 = mxGetN(pa);
    this->bias.resize(dim1, dim2);
    idx = 0;
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim1; j++) {
            this->bias(j, i) = values[idx++];
        }
    }
    // std::cout << this->bias << std::endl;

    pa = matGetVariable(pmFile, rm_name.c_str());
    values = (float_t *) mxGetData(pa);
    dim1 = mxGetM(pa);
    dim2 = mxGetN(pa);
    this->running_mean.resize(dim1, dim2);
    idx = 0;
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim1; j++) {
            this->running_mean(j, i) = values[idx++];
        }
    }
    // std::cout << this->running_mean << std::endl;

    pa = matGetVariable(pmFile, rv_name.c_str());
    values = (float_t *) mxGetData(pa);
    dim1 = mxGetM(pa);
    dim2 = mxGetN(pa);
    this->running_var.resize(dim1, dim2);
    idx = 0;
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim1; j++) {
            this->running_var(j, i) = values[idx++];
        }
    }
    // std::cout << this->running_var << std::endl;

    pa = matGetVariable(pmFile, nbt_name.c_str());
    auto nbt_value = (int32_t *) mxGetData(pa);
    this->num_batches_tracked = nbt_value[0];
    // std::cout << this->num_batches_tracked << std::endl;
}

void Layer_BatchNorm2d::LoadTestState() {
    Eigen::Tensor<float_t, 2> w(1, this->channels);
    Eigen::Tensor<float_t, 2> b(1, this->channels);
    Eigen::Tensor<float_t, 2> rm(1, this->channels);
    Eigen::Tensor<float_t, 2> rv(1, this->channels);
    w.setConstant(1);
    b.setConstant(0);
    rm.setConstant(1);
    rv.setConstant(2);
    this->weights = w;
    this->bias = b;
    this->running_mean = rm;
    this->running_var = rv;
}


Eigen::Tensor<float_t, 4> Layer_BatchNorm2d::forward(Eigen::Tensor<float_t, 4> &input) {
    int64_t N_CHANNEL = this->channels;
    const Eigen::Tensor<float_t, 4>::Dimensions &dim_inp = input.dimensions();
    Eigen::Tensor<float_t, 4> output(dim_inp);
    Eigen::Tensor<float_t, 3> cur_channel(dim_inp[0], dim_inp[2], dim_inp[3]);
    Eigen::Tensor<float_t, 3> cur_res(dim_inp[0], dim_inp[2], dim_inp[3]);
    Eigen::Tensor<float_t, 3> cur_w(dim_inp[0], dim_inp[2], dim_inp[3]);
    Eigen::Tensor<float_t, 3> cur_b(dim_inp[0], dim_inp[2], dim_inp[3]);
    Eigen::Tensor<float_t, 3> cur_mean(dim_inp[0], dim_inp[2], dim_inp[3]);
    Eigen::Tensor<float_t, 3> cur_var(dim_inp[0], dim_inp[2], dim_inp[3]);
    for (int c = 0; c < N_CHANNEL; c++) {
        cur_channel = input.chip(c, 1);
        cur_w.setConstant(this->weights(0, c));
        cur_b.setConstant(this->bias(0, c));
        cur_mean.setConstant(this->running_mean(0, c));
        cur_var.setConstant(this->running_var(0, c));
        cur_res = (cur_channel - cur_mean) / cur_var.pow(0.5) * cur_w + cur_b;
        output.chip(c, 1) = cur_res;
    }
    return output;
}

