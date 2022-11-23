//
// Created by Koer on 2022/10/31.
//

#include "../include/NonLinearity.h"

NonLinearity::NonLinearity() {

}

Eigen::Tensor<float_t, 4> NonLinearity::ELU(Eigen::Tensor<float_t, 4> &input, float_t alpha) {
    const Eigen::Tensor<double, 4>::Dimensions &d = input.dimensions();
    Eigen::Tensor<float_t, 4> output(d);
    for (int i = 0; i < d[0]; i++) {
        for (int j = 0; j < d[1]; j++) {
            for (int k = 0; k < d[2]; k++) {
                for (int l = 0; l < d[3]; l++) {
                    if (input(i, j, k, l) <= 0) {
                        output(i, j, k, l) = alpha * exp(input(i, j, k, l)) - 1;
                    } else {
                        output(i, j, k, l) = input(i, j, k, l);
                    }
                }
            }
        }
    }
    return output;
}

Eigen::Tensor<float_t, 4> NonLinearity::ReLU(Eigen::Tensor<float_t, 4> &input) {
    const Eigen::Tensor<double, 4>::Dimensions &d = input.dimensions();
    Eigen::Tensor<float_t, 4> output(d);
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
    return output;
}

Eigen::Tensor<float_t, 4> NonLinearity::Softplus(Eigen::Tensor<float_t, 4> &input, float_t beta, float_t threshold) {
    const Eigen::Tensor<double, 4>::Dimensions &d = input.dimensions();
    Eigen::Tensor<float_t, 4> output(d);
    for (int i = 0; i < d[0]; i++) {
        for (int j = 0; j < d[1]; j++) {
            for (int k = 0; k < d[2]; k++) {
                for (int l = 0; l < d[3]; l++) {
                    if (input(i, j, k, l) * beta <= threshold) {
                        output(i, j, k, l) = logf(1 + expf((input(i, j, k, l) * beta)) / beta);
                    }
                }
            }
        }
    }
    return output;
}







