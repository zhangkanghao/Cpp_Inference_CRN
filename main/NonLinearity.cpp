//
// Created by Koer on 2022/10/31.
//

#include "../include/NonLinearity.h"

NonLinearity::NonLinearity() {
    this->type = "ReLU";
}

NonLinearity::NonLinearity(std::string type, bool inplace) {
    if (strcmp(type.c_str(), "ReLU") != 0)
        throw "NonLinearity Type Error, choose from ELU, ReLU, PReLU, Softplus";
    this->type = type;
    this->inplace = inplace;
}

NonLinearity::NonLinearity(std::string type, int32_t val, bool inplace) {
    if (strcmp(type.c_str(), "PReLU") != 0)
        throw "NonLinearity Type Error, choose from ELU, ReLU, PReLU, Softplus";
    this->type = type;
    this->channels = val;
    this->inplace = inplace;
}

NonLinearity::NonLinearity(std::string type, float_t val, bool inplace) {
    if (strcmp(type.c_str(), "ELU") != 0)
        throw "NonLinearity Type Error, choose from ELU, ReLU, PReLU, Softplus";
    this->type = type;
    this->alpha = val;
    this->inplace = inplace;
}

NonLinearity::NonLinearity(std::string type, float_t beta, float_t threshold) {
    if (strcmp(type.c_str(), "Softplus") != 0)
        throw "NonLinearity Type Error, choose from ELU, ReLU, PReLU, Softplus";
    this->type = type;
    this->beta = beta;
    this->threshold = threshold;
}


void NonLinearity::ELU(Eigen::Tensor<float_t, 4> &input, Eigen::Tensor<float_t, 4> &output) {
    const Eigen::Tensor<double, 4>::Dimensions &d = input.dimensions();
    for (int i = 0; i < d[0]; i++) {
        for (int j = 0; j < d[1]; j++) {
            for (int k = 0; k < d[2]; k++) {
                for (int l = 0; l < d[3]; l++) {
                    if (input(i, j, k, l) <= 0) {
                        output(i, j, k, l) = this->alpha *
                                            exp(input(i, j, k, l) - 1);
                    }
                }
            }
        }
    }
}

void NonLinearity::ReLU(Eigen::Tensor<float_t, 4> &input, Eigen::Tensor<float_t, 4> &output) {
    const Eigen::Tensor<double, 4>::Dimensions &d = input.dimensions();
    for (int i = 0; i < d[0]; i++) {
        for (int j = 0; j < d[1]; j++) {
            for (int k = 0; k < d[2]; k++) {
                for (int l = 0; l < d[3]; l++) {
                    if (input(i, j, k, l) <= 0) {
                        output(i, j, k, l) = 0.0;
                    }
                }
            }
        }
    }
}

void NonLinearity::PReLU(Eigen::Tensor<float_t, 4> &input, Eigen::Tensor<float_t, 4> &output) {

}

void NonLinearity::Softplus(Eigen::Tensor<float_t, 4> &input, Eigen::Tensor<float_t, 4> &output) {
    const Eigen::Tensor<double, 4>::Dimensions &d = input.dimensions();
    for (int i = 0; i < d[0]; i++) {
        for (int j = 0; j < d[1]; j++) {
            for (int k = 0; k < d[2]; k++) {
                for (int l = 0; l < d[3]; l++) {
                    if (input(i, j, k, l) * this->beta <= this->threshold) {
                        output(i, j, k, l) = logf(1 + expf((input(i, j, k, l) * this->beta)) / this->beta);
                    }
                }
            }
        }
    }
}

void NonLinearity::Sigmoid(Eigen::Tensor<float_t, 4> &input, Eigen::Tensor<float_t, 4> &output) {
    output = input.sigmoid();
}







