//
// Created by Koer on 2022/10/31.
//

#include "iostream"
#include "../include/Layer_TransposedConv2d.h"

Layer_TransposedConv2d::Layer_TransposedConv2d() {
    this->in_channels = 1;
    this->out_channels = 1;
    this->kernel_size = std::make_pair(1, 1);
    this->stride = std::make_pair(1, 1);
    this->padding = std::make_pair(0, 0);
    this->out_padding = std::make_pair(0, 0);
}

Layer_TransposedConv2d::Layer_TransposedConv2d(int64_t in_ch, int64_t out_ch, std::pair<int64_t, int64_t> kernel,
                                               std::pair<int64_t, int64_t> stride, std::pair<int64_t, int64_t> dilation,
                                               std::pair<int64_t, int64_t> padding,
                                               std::pair<int64_t, int64_t> out_padding) {
    /* code */
    this->in_channels = in_ch;
    this->out_channels = out_ch;
    this->kernel_size = kernel;
    this->stride = stride;
    this->dilation = dilation;
    this->padding = padding;
    this->out_padding = out_padding;
}

void Layer_TransposedConv2d::LoadState(MATFile *pmFile, std::string state_preffix) {
    std::string weight_name = state_preffix + "_weight";
    std::string bias_name = state_preffix + "_bias";

    // Read weight
    mxArray *pa = matGetVariable(pmFile, weight_name.c_str());
    auto *values = (float_t *) mxGetData(pa);
    // First Dimension  eg.(16,1,2,3)  ===> M=16
    long long dim1 = mxGetM(pa);
    // Rest Total Dimension eg.(16,1,2,3) ===>N = 1 * 2 * 3 = 6
    long long dim2 = mxGetN(pa);
    dim2 = dim2 / this->kernel_size.first / this->kernel_size.second;
    this->weights.resize(dim1, dim2, this->kernel_size.first, this->kernel_size.second);
    int idx = 0;
    for (int i = 0; i < this->kernel_size.second; i++) {
        for (int j = 0; j < this->kernel_size.first; j++) {
            for (int k = 0; k < dim2; k++) {
                for (int l = 0; l < dim1; l++) {
                    this->weights(l, k, j, i) = values[idx++];
                }
            }
        }
    }
    // std::cout << this->weights << std::endl;

    // Read bias
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
    std::cout << " Finish Loading State of " + state_preffix << std::endl;
}

void Layer_TransposedConv2d::LoadTestState() {
    this->weights(this->out_channels, this->in_channels, this->kernel_size.first, this->kernel_size.second);
    this->weights.setConstant(1.0);
    this->bias(1, this->out_channels);
    this->bias.setConstant(0.0);
}

Eigen::Tensor<float_t, 4> Layer_TransposedConv2d::forward(Eigen::Tensor<float_t, 4> &input) {
    const Eigen::Tensor<size_t, 4>::Dimensions &dim_inp = input.dimensions();
    std::pair<int64_t, int64_t> trans_padding = std::make_pair(this->kernel_size.first - 1 - this->padding.first,
                                                               this->kernel_size.second - 1 - this->padding.second);
    // pad大小包含在输入每个点之间插入strid-1个0，在首尾各插入pad个0，以及在首尾各插入(K-1)*(D-1)个0
    int64_t pad_size_time = trans_padding.first + (this->kernel_size.first - 1) * (this->dilation.first - 1);
    int64_t pad_size_freq = trans_padding.second + (this->kernel_size.second - 1) * (this->dilation.second - 1);
    int64_t batch = dim_inp[0], C_in = dim_inp[1], H_in = dim_inp[2], W_in = dim_inp[3];
    int64_t H_pad = H_in + pad_size_time * 2 + (H_in - 1) * (this->stride.first - 1);
    int64_t W_pad = W_in + pad_size_freq * 2 + (W_in - 1) * (this->stride.second - 1);

    /* padding tensor */
    Eigen::Tensor<float_t, 4> padded_input = Eigen::Tensor<float_t, 4>(batch, C_in, H_pad, W_pad);
    padded_input.setZero();
    padded_input.stridedSlice(
            Eigen::array<int64_t, 4>{0, 0, pad_size_time, pad_size_freq},
            Eigen::array<int64_t, 4>{batch, this->in_channels, H_pad - pad_size_time, W_pad - pad_size_freq},
            Eigen::array<int64_t, 4>{1, 1, this->stride.first, this->stride.second}) = input;

    /* output shape */
    int64_t H_out = (H_in - 1) * this->stride.first - 2 * this->padding.first +
                    this->dilation.first * (this->kernel_size.first - 1) +
                    this->out_padding.first + 1;
    int64_t W_out = (H_in - 1) * this->stride.second - 2 * this->padding.second +
                    this->dilation.second * (this->kernel_size.second - 1) +
                    this->out_padding.second + 1;
    Eigen::Tensor<float_t, 4> output = Eigen::Tensor<float_t, 4>(batch, out_channels, H_out, W_out);
    output.setZero();


    Eigen::Tensor<float_t, 4> region;
    Eigen::Tensor<float_t, 4> kernel;
    Eigen::Tensor<float_t, 1> tmp_res;
    Eigen::array<int, 3> dim_sum{1, 2, 3};
    int64_t h_region = (this->kernel_size.first - 1) * this->dilation.first;
    int64_t w_region = (this->kernel_size.second - 1) * this->dilation.second;
    for (int64_t idx_batch = 0; idx_batch < batch; idx_batch++) {
        for (int64_t idx_outc = 0; idx_outc < this->out_channels; idx_outc++) {
            kernel = this->weights.slice(Eigen::array<int64_t, 4>{idx_outc, 0, 0, 0},
                                         Eigen::array<int64_t, 4>{1, this->in_channels, this->kernel_size.first,
                                                                  this->kernel_size.second}
            );
            for (int64_t idx_h = 0; idx_h < H_pad - h_region; idx_h++) {
                for (int64_t idx_w = 0; idx_w < W_pad - w_region; idx_w++) {
                    region = padded_input.stridedSlice(
                            Eigen::array<int64_t, 4>{idx_batch, 0, idx_h, idx_w},
                            Eigen::array<int64_t, 4>{idx_batch + 1, this->in_channels, idx_h + h_region + 1,
                                                     idx_w + w_region + 1},
                            Eigen::array<int64_t, 4>{1, 1, this->dilation.first, this->dilation.second});
                    tmp_res = (region * kernel).sum(dim_sum);
                    output(idx_batch, idx_outc, idx_h, idx_w) = tmp_res(0) + this->bias(0, idx_outc);
                }
            }
        }
    }
    /* set out_padding value, rows = bias, cols = cols[-1] */

    if (this->out_padding.first > 0 && this->out_padding.first < this->stride.first) {
        for (int64_t idx_batch = 0; idx_batch < batch; idx_batch++) {
            for (int64_t idx_outc = 0; idx_outc < this->out_channels; idx_outc++) {
                for (int64_t idx_h = H_out - this->out_padding.first; idx_h < H_out; idx_h++) {
                    for (int64_t idx_w = 0; idx_w < W_out; idx_w++) {
                        output(idx_batch, idx_outc, idx_h, idx_w) = this->bias(0, idx_outc);
                    }
                }
            }
        }
    }
    if (this->out_padding.second > 0 && this->out_padding.second < this->stride.second) {
        for (int64_t idx_batch = 0; idx_batch < batch; idx_batch++) {
            for (int64_t idx_outc = 0; idx_outc < this->out_channels; idx_outc++) {
                for (int64_t idx_h = 0; idx_h < H_out; idx_h++) {
                    for (int64_t idx_w = W_out - this->out_padding.second; idx_w < W_out; idx_w++) {
                        output(idx_batch, idx_outc, idx_h, idx_w) = output(idx_batch, idx_outc, idx_h, idx_w - 1);
                    }
                }
            }
        }
    }

    return output;
}


