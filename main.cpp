#include <iostream>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "include/Wav_File.h"
#include "include/Model_CRN.h"
#include "unsupported/Eigen/CXX11/Tensor"

using namespace std;

void print(Eigen::Tensor<float_t, 4> input) {
    const Eigen::Tensor<size_t, 4>::Dimensions &dim_inp = input.dimensions();
    cout << "Variable:" << endl;
    cout << "[";
    for (int i = 0; i < dim_inp[0]; i++) {
        if (i > 0) {
            cout << " ";
        }
        cout << "[";
        for (int j = 0; j < dim_inp[1]; j++) {
            if (j > 0) {
                cout << "  ";
            }
            cout << "[";
            for (int k = 0; k < dim_inp[2]; k++) {
                if (k > 0) {
                    cout << "   ";
                }
                cout << "[";
                for (int l = 0; l < dim_inp[3]; l++) {
                    cout << input(i, j, k, l);
                    if (l < dim_inp[3] - 1) {
                        cout << "\t";
                    }
                }
                cout << "]";
                if (k < dim_inp[2] - 1) {
                    cout << "," << endl;
                }
            }
            cout << "]";
            if (j < dim_inp[1] - 1) {
                cout << endl << endl;
            }
        }
        cout << "]";
        if (i < dim_inp[0] - 1) {
            cout << endl;
        }
    }
    cout << "]" << endl;
}


/*
// Test Convolution
int32_t out_channels = 1, in_channels = 1;
std::pair<int32_t, int32_t> kernel_size = make_pair(3   , 3);
std::pair<int32_t, int32_t> stride = make_pair(1, 2);
std::pair<int32_t, int32_t> dilation = make_pair(1, 2);
std::pair<int32_t, int32_t> padding = make_pair(0, 1);
std::pair<int32_t, int32_t> out_padding = make_pair(0, 0);



Eigen::Tensor<float_t, 4> forward(Eigen::Tensor<float_t, 4> &input) {
    Eigen::Tensor<float_t, 4> weight(out_channels, in_channels, 3, 3);
    weight.setConstant(1);
    print(weight);

    Eigen::Tensor<float_t, 2> bias(1, out_channels);
    bias.setValues({{1.0f},
                    {2.0f}});
    cout << "bias:";
    cout << bias << endl;


    const Eigen::Tensor<int32_t, 4>::Dimensions &dim_inp = input.dimensions();

    // Sequence channel × T × F
    int32_t pad_size_time = padding.first;
    int32_t pad_size_freq = padding.second;
    int32_t batch = dim_inp[0], C_in = dim_inp[1], H_in = dim_inp[2], W_in = dim_inp[3];
    int32_t H_pad = H_in + pad_size_time * 2;
    int32_t W_pad = W_in + pad_size_freq * 2;

    // padding tensor
    Eigen::Tensor<float_t, 4> padded_input = Eigen::Tensor<float_t, 4>(batch, C_in, H_pad, W_pad);
    const Eigen::Tensor<int32_t, 4>::Dimensions &dim_pad = padded_input.dimensions();
    padded_input.setZero();
    padded_input.slice(Eigen::array<int32_t, 4>{0, 0, pad_size_time, pad_size_freq}, dim_inp) = input;
    print(padded_input);

    // output shape
    int32_t H_out = (H_pad - dilation.first * (kernel_size.first - 1) - 1) / stride.first + 1;
    int32_t W_out = (W_pad - dilation.second * (kernel_size.second - 1) - 1) / stride.second + 1;
    Eigen::Tensor<float_t, 4> output(batch, out_channels, H_out, W_out);
    output.setZero();

    // params
//     * region: tmp storage of map to be convolved
//     * kernel: tmp storage of kernel of the out_channels idx_outc
//     * tmp_res: tmp storage of res (convolve all in_channels and sum up)
//     * dim_sum: the origin tmp_res is at view of (1,ic,k1,k2), sum along the 1,2,3 axis
//     * h_region: the h of convolve region - 1
//     * w_region: the w of convolve region - 1

    Eigen::Tensor<float_t, 4> region;
    Eigen::Tensor<float_t, 4> kernel;
    Eigen::Tensor<float_t, 1> tmp_res;
    Eigen::array<int, 3> dim_sum{1, 2, 3};
    int32_t h_region = (kernel_size.first - 1) * dilation.first;
    int32_t w_region = (kernel_size.second - 1) * dilation.second;
    for (int32_t idx_batch = 0; idx_batch < dim_pad[0]; idx_batch++) {
        for (int32_t idx_outc = 0; idx_outc < out_channels; idx_outc++) {
            kernel = weight.slice(Eigen::array<int32_t, 4>{idx_outc, 0, 0, 0},
                                  Eigen::array<int32_t, 4>{1, in_channels, kernel_size.first, kernel_size.second}
            );
            for (int32_t idx_h = 0; idx_h < dim_pad[2] - h_region; idx_h += stride.first) {
                for (int32_t idx_w = 0; idx_w < dim_pad[3] - w_region; idx_w += stride.second) {
                    region = padded_input.stridedSlice(
                            Eigen::array<int64_t, 4>{idx_batch, 0, idx_h, idx_w},
                            Eigen::array<int64_t, 4>{idx_batch + 1, in_channels, idx_h + h_region + 1,
                                                     idx_w + w_region + 1},
                            Eigen::array<int64_t, 4>{1, 1, dilation.first, dilation.second});
                    print(region);
                    tmp_res = (region * kernel).sum(dim_sum);
                    cout << tmp_res << endl;
                    output(idx_batch, idx_outc, idx_h / stride.first, idx_w / stride.second) =
                            tmp_res(0) + bias(0, idx_outc);
                }
            }
        }
    }
    return output;
}

Eigen::Tensor<float_t, 4> tranposed_conv2d(Eigen::Tensor<float_t, 4> &input) {
    Eigen::Tensor<float_t, 4> weights(out_channels, in_channels, 3, 3);
    weights.setConstant(1);
    print(weights);
    Eigen::Tensor<float_t, 2> bias(1, out_channels);
    bias.setValues({{0.0f},
                    {0.0f}});
    cout << "bias:";
    cout << bias << endl;

    const Eigen::Tensor<size_t, 4>::Dimensions &dim_inp = input.dimensions();

    // Sequence channel × T × F
    std::pair<int16_t, int16_t> trans_padding = std::make_pair(kernel_size.first - 1 - padding.first,
                                                               kernel_size.second - 1 - padding.second);
    // pad大小包含在输入每个点之间插入strid-1个0，在首尾各插入pad个0，以及在首尾各插入(K-1)*(D-1)个0
    int64_t pad_size_time = trans_padding.first + (kernel_size.first - 1) * (dilation.first - 1);
    int64_t pad_size_freq = trans_padding.second + (kernel_size.second - 1) * (dilation.second - 1);
    int64_t batch = dim_inp[0], C_in = dim_inp[1], H_in = dim_inp[2], W_in = dim_inp[3];
    int64_t H_pad = H_in + pad_size_time * 2 + (H_in - 1) * (stride.first - 1);
    int64_t W_pad = W_in + pad_size_freq * 2 + (W_in - 1) * (stride.second - 1);

    // padding tensor
    Eigen::Tensor<float_t, 4> padded_input = Eigen::Tensor<float_t, 4>(batch, C_in, H_pad, W_pad);
    padded_input.setZero();
    padded_input.stridedSlice(
            Eigen::array<int64_t, 4>{0, 0, pad_size_time, pad_size_freq},
            Eigen::array<int64_t, 4>{batch, in_channels, H_pad - pad_size_time, W_pad - pad_size_freq},
            Eigen::array<int64_t, 4>{1, 1, stride.first, stride.second}) = input;
    print(padded_input);

    // output shape
    int64_t H_out = (H_in - 1) * stride.first - 2 * padding.first + dilation.first * (kernel_size.first - 1) +
                    out_padding.first + 1;
    int64_t W_out = (H_in - 1) * stride.second - 2 * padding.second + dilation.second * (kernel_size.second - 1) +
                    out_padding.second + 1;
    Eigen::Tensor<float_t, 4> output = Eigen::Tensor<float_t, 4>(batch, out_channels, H_out, W_out);
    output.setZero();


    Eigen::Tensor<float_t, 4> region;
    Eigen::Tensor<float_t, 4> kernel;
    Eigen::Tensor<float_t, 1> tmp_res;
    Eigen::array<int, 3> dim_sum{1, 2, 3};
    int32_t h_region = (kernel_size.first - 1) * dilation.first;
    int32_t w_region = (kernel_size.second - 1) * dilation.second;
    for (int32_t idx_batch = 0; idx_batch < batch; idx_batch++) {
        for (int32_t idx_outc = 0; idx_outc < out_channels; idx_outc++) {
            kernel = weights.slice(Eigen::array<int32_t, 4>{idx_outc, 0, 0, 0},
                                   Eigen::array<int32_t, 4>{1, in_channels, kernel_size.first,
                                                            kernel_size.second}
            );
            print(kernel);
            for (int32_t idx_h = 0; idx_h < H_pad - h_region; idx_h++) {
                for (int32_t idx_w = 0; idx_w < W_pad - w_region; idx_w++) {
                    region = padded_input.stridedSlice(
                            Eigen::array<int64_t, 4>{idx_batch, 0, idx_h, idx_w},
                            Eigen::array<int64_t, 4>{idx_batch + 1, in_channels, idx_h + h_region + 1,
                                                     idx_w + w_region + 1},
                            Eigen::array<int64_t, 4>{1, 1, dilation.first, dilation.second});
                    print(region);
                    tmp_res = (region * kernel).sum(dim_sum);
                    output(idx_batch, idx_outc, idx_h, idx_w) = tmp_res(0) + bias(0, idx_outc);
                }
            }
        }
    }
    print(output);
    // set out_padding value, rows = bias, cols = cols[-1]
    if (out_padding.first > 0) {
        for (int64_t idx_batch = 0; idx_batch < batch; idx_batch++) {
            for (int64_t idx_outc = 0; idx_outc < out_channels; idx_outc++) {
                for (int64_t idx_h = H_out - out_padding.first; idx_h < H_out; idx_h++) {
                    for (int64_t idx_w = 0; idx_w < W_out; idx_w++) {
                        output(idx_batch, idx_outc, idx_h, idx_w) = bias(0, idx_outc);
                    }
                }
            }
        }
    }
    if (out_padding.second > 0) {
        for (int64_t idx_batch = 0; idx_batch < batch; idx_batch++) {
            for (int64_t idx_outc = 0; idx_outc < out_channels; idx_outc++) {
                for (int64_t idx_h = 0; idx_h < H_out; idx_h++) {
                    for (int64_t idx_w = W_out - out_padding.second; idx_w < W_out; idx_w++) {
                        output(idx_batch, idx_outc, idx_h, idx_w) = output(idx_batch, idx_outc, idx_h, idx_w - 1);
                    }
                }
            }
        }
    }
    print(output);

    return output;
}
*/
Eigen::Tensor<float_t, 4> transpose(Eigen::Tensor<float_t, 4> &input, Eigen::array<int64_t, 4> trans_idx) {
    const Eigen::Tensor<size_t, 4>::Dimensions &dim_inp = input.dimensions();
    Eigen::Tensor<size_t, 4>::Dimensions dim_out;
    for (int i = 0; i < 4; i++) {
        dim_out[i] = dim_inp[trans_idx[i]];
    }

    Eigen::Tensor<float_t, 4> output(dim_out[0], dim_out[1], dim_out[2], dim_out[3]);
    for (int64_t i = 0; i < dim_out[0]; i++) {
        for (int64_t j = 0; j < dim_out[1]; j++) {
            for (int64_t k = 0; k < dim_out[2]; k++) {
                for (int64_t l = 0; l < dim_out[3]; l++) {
                    int64_t idx_inp[4] = {i, j, k, l};
                    output(i, j, k, l) = input(idx_inp[trans_idx[0]], idx_inp[trans_idx[1]], idx_inp[trans_idx[2]],
                                               idx_inp[trans_idx[3]]);
                }
            }
        }
    }
    return output;
}

Eigen::Tensor<float_t, 3> viewForward(Eigen::Tensor<float_t, 4> &input) {
    const Eigen::Tensor<size_t, 4>::Dimensions &dim_inp = input.dimensions();

    Eigen::Tensor<float_t, 3> output(dim_inp[0], dim_inp[1], dim_inp[2] * dim_inp[3]);
    for (int64_t i = 0; i < dim_inp[0]; i++) {
        for (int64_t j = 0; j < dim_inp[1]; j++) {
            for (int64_t k = 0; k < dim_inp[2]; k++) {
                for (int64_t l = 0; l < dim_inp[3]; l++) {
                    output(i, j, k * dim_inp[3] + l) = input(i, j, k, l);
                }
            }
        }
    }
    return output;
}

Eigen::Tensor<float_t, 4> viewBackward(Eigen::Tensor<float_t, 3> &input, Eigen::array<int64_t, 4> dims) {

    Eigen::Tensor<float_t, 4> output(dims[0], dims[1], dims[2], dims[3]);
    for (int64_t i = 0; i < dims[0]; i++) {
        for (int64_t j = 0; j < dims[1]; j++) {
            for (int64_t k = 0; k < dims[2]; k++) {
                for (int64_t l = 0; l < dims[3]; l++) {
                    output(i, j, k, l) = input(i, j, k * dims[3] + l);
                }
            }
        }
    }
    return output;
}


int64_t batch = 2, channel = 2, timel = 3, freq = 2, hidden_size = 6;

Eigen::Tensor<float_t, 3> lstm_forward(Eigen::Tensor<float_t, 3> &input, Eigen::Tensor<float_t, 2> &h_t,
                                       Eigen::Tensor<float_t, 2> &c_t) {

    const Eigen::Tensor<size_t, 3>::Dimensions &dim_inp = input.dimensions();
    int64_t _BATCH = dim_inp[0], _TIME = dim_inp[1], _FREQ = dim_inp[2], _HIDDEN = hidden_size;
    Eigen::Tensor<float_t, 2> weight_ih_l0(_HIDDEN * 4, _FREQ);
    Eigen::Tensor<float_t, 2> weight_hh_l0(_HIDDEN * 4, _HIDDEN);
    Eigen::Tensor<float_t, 2> bias_ih_l0(1, _HIDDEN * 4);
    Eigen::Tensor<float_t, 2> bias_hh_l0(1, _HIDDEN * 4);
    weight_ih_l0.setRandom();
    weight_hh_l0.setRandom();
    bias_ih_l0.setValues(
            {{0.2221f, 0.1316f, 0.1657f, 0.1547f, 0.1213f, 0.8876f, 0.2221f, 0.1316f, 0.1657f, 0.1547f, 0.1213f,
              0.8876f, 0.2221f, 0.1316f, 0.1657f, 0.1547f, 0.1213f, 0.8876f, 0.2221f, 0.1316f, 0.1657f, 0.1547f,
              0.1213f, 0.8876f}});
    bias_hh_l0.setValues(
            {{0.5645f, 0.1556f, 0.7612f, 0.1321f, 0.6545f, 0.4567f, 0.5645f, 0.1556f, 0.7612f, 0.1321f, 0.6545f,
              0.4567f, 0.5645f, 0.1556f, 0.7612f, 0.1321f, 0.6545f, 0.4567f, 0.5645f, 0.1556f, 0.7612f, 0.1321f,
              0.6545f, 0.4567f,}});
    Eigen::Tensor<float_t, 2> bias_ih_broadcast = bias_ih_l0.broadcast(Eigen::array<int64_t, 2>{_BATCH, 1});
    Eigen::Tensor<float_t, 2> bias_hh_broadcast = bias_hh_l0.broadcast(Eigen::array<int64_t, 2>{_BATCH, 1});

    Eigen::Tensor<float_t, 3> output(_BATCH, _TIME, _HIDDEN);
    Eigen::Tensor<float_t, 2> X_t(_BATCH, _FREQ);
    Eigen::Tensor<float_t, 2> gates;
    Eigen::Tensor<float_t, 2> i_t, f_t, g_t, o_t;
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 1)};
    Eigen::array<int64_t, 2> gate_patch = Eigen::array<int64_t, 2>{_BATCH, _HIDDEN};
    for (int t = 0; t < _TIME; t++) {
        X_t = input.chip(t, 1);
        gates = X_t.contract(weight_ih_l0, product_dims) + bias_ih_broadcast +
                h_t.contract(weight_hh_l0, product_dims) + bias_hh_broadcast;
        i_t = gates.slice(Eigen::array<int64_t, 2>{0, _HIDDEN * 0}, gate_patch).sigmoid();
        f_t = gates.slice(Eigen::array<int64_t, 2>{0, _HIDDEN * 1}, gate_patch).sigmoid();
        g_t = gates.slice(Eigen::array<int64_t, 2>{0, _HIDDEN * 2}, gate_patch).tanh();
        o_t = gates.slice(Eigen::array<int64_t, 2>{0, _HIDDEN * 3}, gate_patch).sigmoid();
        c_t = f_t * c_t + i_t * g_t;
        h_t = o_t * c_t.tanh();
        output.chip(t, 1) = h_t;
    }

    return output;
}

int main() {

    Eigen::Tensor<float, 4> input(batch, channel, timel, freq);
    input.setRandom();
    print(input);

    // LSTM
    Eigen::Tensor<float, 4> _in_reshape = transpose(input, Eigen::array<int64_t, 4>{0, 2, 1, 3});
    Eigen::Tensor<float, 3> _in_view = viewForward(_in_reshape);
    cout << _in_view << endl;
    Eigen::Tensor<float, 2> _hidden_state(batch, hidden_size);
    Eigen::Tensor<float, 2> _cell_state(batch, hidden_size);
    _hidden_state.setZero();
    _cell_state.setZero();
    Eigen::Tensor<float, 3> _lstm_out = lstm_forward(_in_view, _hidden_state, _cell_state);
    cout << _lstm_out << endl;
    Eigen::Tensor<float, 4> _out_view = viewBackward(_lstm_out, Eigen::array<int64_t, 4>{batch, timel, channel,
                                                                                         hidden_size / channel});
    print(_out_view);
    Eigen::Tensor<float, 4> _out_reshape = transpose(_out_view, Eigen::array<int64_t, 4>{0, 2, 1, 3});
    print(_out_reshape);

//    const char *path = "C:/Users/65181/CLionProjects/CRN/resources/crn.mat";
//    const char *wav_path = "C:/Users/65181/CLionProjects/CRN/resources/S006_ADTbabble_snr0_tgt.wav";
//    const char *out_path = "C:/Users/65181/CLionProjects/CRN/resources/output_wav.wav";
//    Wav_File wav = Wav_File();
//    wav.LoadWavFile(wav_path);
//    Model_CRN net = Model_CRN(path);
////    net.forward(wav);
//    wav.WriteWavFile(out_path);
//    wav.FreeSource();
//    return 0;

}
