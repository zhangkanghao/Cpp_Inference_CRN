//
// Created by Koer on 2022/10/31.
//
#include "iostream"
#include "windows.h"
#include "vector"
#include "mat.h"
#include "matrix.h"
#include "Layer_Conv2d.h"
#include "Layer_LSTM.h"
#include "Layer_TransposedConv2d.h"
#include "Layer_BatchNorm2d.h"
#include "NonLinearity.h"
#include "Wav_File.h"

#ifndef CRN_MODEL_CRN_H
#define CRN_MODEL_CRN_H


class Model_CRN {
public:
    Model_CRN();

    Model_CRN(const char *state_path);

    void LoadState(const char *state_path);

    void LoadTestState();

    Eigen::Tensor<float_t, 4> forward(Eigen::Tensor<float_t, 4> &inp);

private:
    /* CRN config */
    const std::vector<std::string> layer_name = {"enc_conv1", "enc_conv2", "enc_conv3", "enc_conv4", "enc_conv5",
                                                 "dec_conv1", "dec_conv2", "dec_conv3", "dec_conv4", "dec_conv5",
                                                 "enc_bn1", "enc_bn2", "enc_bn3", "enc_bn4", "enc_bn5",
                                                 "dec_bn1", "dec_bn2", "dec_bn3", "dec_bn4", "dec_bn5",
                                                 "ac", "softplus"};
    const std::vector<int64_t> enc_in_channels_list = {1, 16, 32, 64, 128};
    const std::vector<int64_t> enc_out_channels_list = {16, 32, 64, 128, 256};
    const std::vector<std::pair<int64_t, int64_t>> enc_kernels_list = {5, std::make_pair(1, 3)};
    const std::vector<std::pair<int64_t, int64_t>> enc_strides_list = {5, std::make_pair(1, 2)};
    const std::vector<std::pair<int64_t, int64_t>> enc_dilation_list = {5, std::make_pair(1, 1)};
    const std::vector<std::pair<int64_t, int64_t>> enc_paddings_list = {5, std::make_pair(0, 0)};
    const std::vector<int64_t> dec_in_channels_list = {512, 256, 128, 64, 32};
    const std::vector<int64_t> dec_out_channels_list = {128, 64, 32, 16, 1};
    const std::vector<std::pair<int64_t, int64_t>> dec_kernels_list = {5, std::make_pair(1, 3)};
    const std::vector<std::pair<int64_t, int64_t>> dec_strides_list = {5, std::make_pair(1, 2)};
    const std::vector<std::pair<int64_t, int64_t>> dec_dilation_list = {5, std::make_pair(1, 1)};
    const std::vector<std::pair<int64_t, int64_t>> dec_paddings_list = {5, std::make_pair(0, 0)};
    const std::vector<std::pair<int64_t, int64_t>> dec_outpadds_list = {5, std::make_pair(0, 1)};
    const int64_t lstm_input = 256 * 4;
    const int64_t lstm_hidden = 256 * 4;
    const int64_t lstm_layers = 2;
    const int64_t lstm_direcs = 1;
    const bool lstm_bidirection = false;


    Layer_Conv2d enc_conv1;
    Layer_Conv2d enc_conv2;
    Layer_Conv2d enc_conv3;
    Layer_Conv2d enc_conv4;
    Layer_Conv2d enc_conv5;
    Layer_TransposedConv2d dec_conv5;
    Layer_TransposedConv2d dec_conv4;
    Layer_TransposedConv2d dec_conv3;
    Layer_TransposedConv2d dec_conv2;
    Layer_TransposedConv2d dec_conv1;

    Layer_BatchNorm2d enc_bn1;
    Layer_BatchNorm2d enc_bn2;
    Layer_BatchNorm2d enc_bn3;
    Layer_BatchNorm2d enc_bn4;
    Layer_BatchNorm2d enc_bn5;
    Layer_BatchNorm2d dec_bn5;
    Layer_BatchNorm2d dec_bn4;
    Layer_BatchNorm2d dec_bn3;
    Layer_BatchNorm2d dec_bn2;
    Layer_LSTM lstm;
    NonLinearity ac;


    Eigen::Tensor<float_t, 3> viewForward(Eigen::Tensor<float_t, 4> &input);

    Eigen::Tensor<float_t, 4> viewBackward(Eigen::Tensor<float_t, 3> &input, Eigen::array<int64_t, 4> dims);

    void print(Eigen::Tensor<float_t, 4> input);

    void sprint(Eigen::Tensor<float_t, 4> input);

    void print3(Eigen::Tensor<float_t, 3> input);
};


#endif //CRN_MODEL_CRN_H
