//
// Created by Koer on 2022/10/31.
//

#include "../include/Model_CRN.h"

Model_CRN::Model_CRN() {
    this->enc_conv1 = Layer_Conv2d(this->enc_in_channels_list[0], this->enc_out_channels_list[0],
                                   this->enc_kernels_list[0], this->enc_strides_list[0],
                                   this->enc_paddings_list[0]);
    this->enc_conv2 = Layer_Conv2d(this->enc_in_channels_list[1], this->enc_out_channels_list[1],
                                   this->enc_kernels_list[1], this->enc_strides_list[1],
                                   this->enc_paddings_list[1]);
    this->enc_conv3 = Layer_Conv2d(this->enc_in_channels_list[2], this->enc_out_channels_list[2],
                                   this->enc_kernels_list[2], this->enc_strides_list[2],
                                   this->enc_paddings_list[2]);
    this->enc_conv4 = Layer_Conv2d(this->enc_in_channels_list[3], this->enc_out_channels_list[3],
                                   this->enc_kernels_list[3], this->enc_strides_list[3],
                                   this->enc_paddings_list[3]);
    this->enc_conv5 = Layer_Conv2d(this->enc_in_channels_list[4], this->enc_out_channels_list[4],
                                   this->enc_kernels_list[4], this->enc_strides_list[4],
                                   this->enc_paddings_list[4]);
    this->dec_conv5 = Layer_TransposedConv2d(this->dec_in_channels_list[0], this->dec_out_channels_list[0],
                                             this->dec_kernels_list[0], this->dec_strides_list[0],
                                             this->dec_paddings_list[0]);
    this->dec_conv4 = Layer_TransposedConv2d(this->dec_in_channels_list[1], this->dec_out_channels_list[1],
                                             this->dec_kernels_list[1], this->dec_strides_list[1],
                                             this->dec_paddings_list[1]);
    this->dec_conv3 = Layer_TransposedConv2d(this->dec_in_channels_list[2], this->dec_out_channels_list[2],
                                             this->dec_kernels_list[2], this->dec_strides_list[2],
                                             this->dec_paddings_list[2]);
    this->dec_conv2 = Layer_TransposedConv2d(this->dec_in_channels_list[3], this->dec_out_channels_list[3],
                                             this->dec_kernels_list[3], this->dec_strides_list[3],
                                             this->dec_paddings_list[3]);
    this->dec_conv1 = Layer_TransposedConv2d(this->dec_in_channels_list[4], this->dec_out_channels_list[4],
                                             this->dec_kernels_list[4], this->dec_strides_list[4],
                                             this->dec_paddings_list[4]);
    this->enc_bn1 = Layer_BatchNorm2d(this->enc_out_channels_list[0]);
    this->enc_bn2 = Layer_BatchNorm2d(this->enc_out_channels_list[1]);
    this->enc_bn3 = Layer_BatchNorm2d(this->enc_out_channels_list[2]);
    this->enc_bn4 = Layer_BatchNorm2d(this->enc_out_channels_list[3]);
    this->enc_bn5 = Layer_BatchNorm2d(this->enc_out_channels_list[4]);
    this->dec_bn5 = Layer_BatchNorm2d(this->dec_out_channels_list[0]);
    this->dec_bn4 = Layer_BatchNorm2d(this->dec_out_channels_list[1]);
    this->dec_bn3 = Layer_BatchNorm2d(this->dec_out_channels_list[2]);
    this->dec_bn2 = Layer_BatchNorm2d(this->dec_out_channels_list[3]);
    this->dec_bn1 = Layer_BatchNorm2d(this->dec_out_channels_list[4]);
    this->lstm = Layer_LSTM(this->lstm_param[0], this->lstm_param[1], this->lstm_param[2]);

}

Model_CRN::Model_CRN(const char *state_path) {
    this->enc_conv1 = Layer_Conv2d(this->enc_in_channels_list[0], this->enc_out_channels_list[0],
                                   this->enc_kernels_list[0], this->enc_strides_list[0],
                                   this->enc_paddings_list[0]);
    this->enc_conv2 = Layer_Conv2d(this->enc_in_channels_list[1], this->enc_out_channels_list[1],
                                   this->enc_kernels_list[1], this->enc_strides_list[1],
                                   this->enc_paddings_list[1]);
    this->enc_conv3 = Layer_Conv2d(this->enc_in_channels_list[2], this->enc_out_channels_list[2],
                                   this->enc_kernels_list[2], this->enc_strides_list[2],
                                   this->enc_paddings_list[2]);
    this->enc_conv4 = Layer_Conv2d(this->enc_in_channels_list[3], this->enc_out_channels_list[3],
                                   this->enc_kernels_list[3], this->enc_strides_list[3],
                                   this->enc_paddings_list[3]);
    this->enc_conv5 = Layer_Conv2d(this->enc_in_channels_list[4], this->enc_out_channels_list[4],
                                   this->enc_kernels_list[4], this->enc_strides_list[4],
                                   this->enc_paddings_list[4]);
    this->dec_conv5 = Layer_TransposedConv2d(this->dec_in_channels_list[0], this->dec_out_channels_list[0],
                                             this->dec_kernels_list[0], this->dec_strides_list[0],
                                             this->dec_paddings_list[0]);
    this->dec_conv4 = Layer_TransposedConv2d(this->dec_in_channels_list[1], this->dec_out_channels_list[1],
                                             this->dec_kernels_list[1], this->dec_strides_list[1],
                                             this->dec_paddings_list[1]);
    this->dec_conv3 = Layer_TransposedConv2d(this->dec_in_channels_list[2], this->dec_out_channels_list[2],
                                             this->dec_kernels_list[2], this->dec_strides_list[2],
                                             this->dec_paddings_list[2]);
    this->dec_conv2 = Layer_TransposedConv2d(this->dec_in_channels_list[3], this->dec_out_channels_list[3],
                                             this->dec_kernels_list[3], this->dec_strides_list[3],
                                             this->dec_paddings_list[3], std::make_pair(0, 1));
    this->dec_conv1 = Layer_TransposedConv2d(this->dec_in_channels_list[4], this->dec_out_channels_list[4],
                                             this->dec_kernels_list[4], this->dec_strides_list[4],
                                             this->dec_paddings_list[4]);
    this->enc_bn1 = Layer_BatchNorm2d(this->enc_out_channels_list[0]);
    this->enc_bn2 = Layer_BatchNorm2d(this->enc_out_channels_list[1]);
    this->enc_bn3 = Layer_BatchNorm2d(this->enc_out_channels_list[2]);
    this->enc_bn4 = Layer_BatchNorm2d(this->enc_out_channels_list[3]);
    this->enc_bn5 = Layer_BatchNorm2d(this->enc_out_channels_list[4]);
    this->dec_bn5 = Layer_BatchNorm2d(this->dec_out_channels_list[0]);
    this->dec_bn4 = Layer_BatchNorm2d(this->dec_out_channels_list[1]);
    this->dec_bn3 = Layer_BatchNorm2d(this->dec_out_channels_list[2]);
    this->dec_bn2 = Layer_BatchNorm2d(this->dec_out_channels_list[3]);
    this->dec_bn1 = Layer_BatchNorm2d(this->dec_out_channels_list[4]);
    this->lstm = Layer_LSTM(this->lstm_param[0], this->lstm_param[1], this->lstm_param[2]);
    this->LoadState(state_path);
}


void Model_CRN::LoadState(const char *state_path) {
    MATFile *pMatFile;
    pMatFile = matOpen(state_path, "r");

    /* Load State of Model */
    enc_conv1.LoadState(pMatFile, "conv1");
    enc_conv2.LoadState(pMatFile, "conv2");
    enc_conv3.LoadState(pMatFile, "conv3");
    enc_conv4.LoadState(pMatFile, "conv4");
    enc_conv5.LoadState(pMatFile, "conv5");
    dec_conv5.LoadState(pMatFile, "conv5_t");
    dec_conv4.LoadState(pMatFile, "conv4_t");
    dec_conv3.LoadState(pMatFile, "conv3_t");
    dec_conv2.LoadState(pMatFile, "conv2_t");
    dec_conv1.LoadState(pMatFile, "conv1_t");
    enc_bn1.LoadState(pMatFile, "bn1");
    enc_bn2.LoadState(pMatFile, "bn2");
    enc_bn3.LoadState(pMatFile, "bn3");
    enc_bn4.LoadState(pMatFile, "bn4");
    enc_bn5.LoadState(pMatFile, "bn5");
    dec_bn5.LoadState(pMatFile, "bn5_t");
    dec_bn4.LoadState(pMatFile, "bn4_t");
    dec_bn3.LoadState(pMatFile, "bn3_t");
    dec_bn2.LoadState(pMatFile, "bn2_t");
    dec_bn1.LoadState(pMatFile, "bn1_t");
    lstm.LoadState(pMatFile, "lstm");

    matClose(pMatFile);

}

void Model_CRN::forward(Wav_File &input) {
    /* 语音预处理 */
    input.setSTFT(512, 256, "hamming");
    input.STFT();
    input.getMagnitude();
    input.getPhase();

    /* CRN运算 */


    input.magToSpec();
    input.ISTFT();

}

Eigen::Tensor<float_t, 3> Model_CRN::viewForward(Eigen::Tensor<float_t, 4> &input) {
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


