//
// Created by Koer on 2022/10/31.
//
#include "iostream"
#include "../include/Layer_BatchNorm2d.h"

Layer_BatchNorm2d::Layer_BatchNorm2d() {
    this->channels = 1;
}

Layer_BatchNorm2d::Layer_BatchNorm2d(int16_t bn_ch) {
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

void Layer_BatchNorm2d::forward(Eigen::Tensor<float_t, 4> &input) {

}
