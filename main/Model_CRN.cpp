//
// Created by Koer on 2022/10/31.
//

#include "../include/Model_CRN.h"

Model_CRN::Model_CRN() {
    this->enc_conv1 = Layer_Conv2d(this->enc_in_channels_list[0], this->enc_out_channels_list[0],
                                   this->enc_kernels_list[0], this->enc_strides_list[0],
                                   this->enc_dilation_list[0], this->enc_paddings_list[0]);
    this->enc_conv2 = Layer_Conv2d(this->enc_in_channels_list[1], this->enc_out_channels_list[1],
                                   this->enc_kernels_list[1], this->enc_strides_list[1],
                                   this->enc_dilation_list[1], this->enc_paddings_list[1]);
    this->enc_conv3 = Layer_Conv2d(this->enc_in_channels_list[2], this->enc_out_channels_list[2],
                                   this->enc_kernels_list[2], this->enc_strides_list[2],
                                   this->enc_dilation_list[2], this->enc_paddings_list[2]);
    this->enc_conv4 = Layer_Conv2d(this->enc_in_channels_list[3], this->enc_out_channels_list[3],
                                   this->enc_kernels_list[3], this->enc_strides_list[3],
                                   this->enc_dilation_list[3], this->enc_paddings_list[3]);
    this->enc_conv5 = Layer_Conv2d(this->enc_in_channels_list[4], this->enc_out_channels_list[4],
                                   this->enc_kernels_list[4], this->enc_strides_list[4],
                                   this->enc_dilation_list[4], this->enc_paddings_list[4]);
    this->dec_conv5 = Layer_TransposedConv2d(this->dec_in_channels_list[0], this->dec_out_channels_list[0],
                                             this->dec_kernels_list[0], this->dec_strides_list[0],
                                             this->dec_dilation_list[0], this->dec_paddings_list[0]);
    this->dec_conv4 = Layer_TransposedConv2d(this->dec_in_channels_list[1], this->dec_out_channels_list[1],
                                             this->dec_kernels_list[1], this->dec_strides_list[1],
                                             this->dec_dilation_list[1], this->dec_paddings_list[1]);
    this->dec_conv3 = Layer_TransposedConv2d(this->dec_in_channels_list[2], this->dec_out_channels_list[2],
                                             this->dec_kernels_list[2], this->dec_strides_list[2],
                                             this->dec_dilation_list[2], this->dec_paddings_list[2]);
    this->dec_conv2 = Layer_TransposedConv2d(this->dec_in_channels_list[3], this->dec_out_channels_list[3],
                                             this->dec_kernels_list[3], this->dec_strides_list[3],
                                             this->dec_dilation_list[3], this->dec_paddings_list[3],
                                             this->dec_outpadds_list[3]);
    this->dec_conv1 = Layer_TransposedConv2d(this->dec_in_channels_list[4], this->dec_out_channels_list[4],
                                             this->dec_kernels_list[4], this->dec_strides_list[4],
                                             this->dec_dilation_list[4], this->dec_paddings_list[4]);
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
    this->lstm = Layer_LSTM(this->lstm_input, this->lstm_hidden, this->lstm_layers, this->lstm_bidirection);
    this->ac = NonLinearity();


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
    this->lstm = Layer_LSTM(this->lstm_input, this->lstm_hidden, this->lstm_layers, this->lstm_bidirection);
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

void Model_CRN::LoadTestState() {
    // w=1,b=0
    enc_conv1.LoadTestState();
    enc_conv2.LoadTestState();
    enc_conv3.LoadTestState();
    enc_conv4.LoadTestState();
    enc_conv5.LoadTestState();
    dec_conv5.LoadTestState();
    dec_conv4.LoadTestState();
    dec_conv3.LoadTestState();
    dec_conv2.LoadTestState();
    dec_conv1.LoadTestState();
    // w=1, b=0, rm=1, rv=2
    enc_bn1.LoadTestState();
    enc_bn2.LoadTestState();
    enc_bn3.LoadTestState();
    enc_bn4.LoadTestState();
    enc_bn5.LoadTestState();
    dec_bn5.LoadTestState();
    dec_bn4.LoadTestState();
    dec_bn3.LoadTestState();
    dec_bn2.LoadTestState();
    dec_bn1.LoadTestState();
    // wih=2, whh=2, bih = 1.0, bhh=1.0
    lstm.LoadTestState();

}

void Model_CRN::print(Eigen::Tensor<float_t, 4> input) {
    const Eigen::Tensor<size_t, 4>::Dimensions &dim_inp = input.dimensions();
    std::cout << "Variable:" << std::endl;
    std::cout << "[";
    for (int i = 0; i < dim_inp[0]; i++) {
        if (i > 0) {
            std::cout << " ";
        }
        std::cout << "[";
        for (int j = 0; j < dim_inp[1]; j++) {
            if (j > 0) {
                std::cout << "  ";
            }
            std::cout << "[";
            for (int k = 0; k < dim_inp[2]; k++) {
                if (k > 0) {
                    std::cout << "   ";
                }
                std::cout << "[";
                for (int l = 0; l < dim_inp[3]; l++) {
                    std::cout << input(i, j, k, l);
                    if (l < dim_inp[3] - 1) {
                        std::cout << "\t";
                    }
                }
                std::cout << "]";
                if (k < dim_inp[2] - 1) {
                    std::cout << "," << std::endl;
                }
            }
            std::cout << "]";
            if (j < dim_inp[1] - 1) {
                std::cout << std::endl << std::endl;
            }
        }
        std::cout << "]";
        if (i < dim_inp[0] - 1) {
            std::cout << std::endl;
        }
    }
    std::cout << "]" << std::endl;
}


void Model_CRN::forward() {
    Eigen::Tensor<float_t, 4> inp(1, 1, 3, 9);
    inp.setRandom();
    print(inp);

    Eigen::Tensor<float_t, 4> e1 = this->enc_conv1.forward(inp);
    e1 = this->enc_bn1.forward(e1);
    e1 = this->ac.ELU(e1);
    print(e1);
    Eigen::Tensor<float_t, 4> e2 = this->enc_conv2.forward(e1);
    e2 = this->enc_bn2.forward(e2);
    e2 = this->ac.ELU(e2);
    print(e2);
    Eigen::Tensor<float_t, 4> e3 = this->enc_conv3.forward(e2);
    e3 = this->enc_bn3.forward(e3);
    e3 = this->ac.ELU(e3);
    Eigen::Tensor<float_t, 4> e4 = this->enc_conv4.forward(e3);
    e4 = this->enc_bn4.forward(e4);
    e4 = this->ac.ELU(e4);
    Eigen::Tensor<float_t, 4> e5 = this->enc_conv5.forward(e4);
    e5 = this->enc_bn5.forward(e5);
    e5 = this->ac.ELU(e5);

    Eigen::Tensor<float_t, 4>::Dimensions dims = e5.dimensions();
    Eigen::array<int64_t, 4> shuffling{0, 2, 1, 3};
    Eigen::Tensor<float_t, 4> lstm_shuffle = e2.shuffle(shuffling);
    Eigen::Tensor<float_t, 3> lstm_in = this->viewForward(lstm_shuffle);
    std::vector<Eigen::Tensor<float_t, 2>> h_t;
    std::vector<Eigen::Tensor<float_t, 2>> c_t;
    Eigen::Tensor<float_t, 3> lstm_out = this->lstm.forward(lstm_in, h_t, c_t);
    Eigen::array<int64_t, 4> lstm_out_shape{dims[0], dims[2], dims[1], this->lstm_hidden * this->lstm_direcs / dims[1]};
    Eigen::Tensor<float_t, 4> lstm_out_view = this->viewBackward(lstm_out, lstm_out_shape);
    Eigen::Tensor<float_t, 4> lstm_out_shuffle = lstm_out_view.shuffle(shuffling);
    print(lstm_out_shuffle);

    Eigen::Tensor<float_t, 4> d5_cat = lstm_out_shuffle.concatenate(e5, 1);
    Eigen::Tensor<float_t, 4> d5 = this->dec_conv5.forward(d5_cat);
    d5 = this->dec_bn5.forward(d5);
    d5 = this->ac.ELU(d5);
    Eigen::Tensor<float_t, 4> d4_cat = d5.concatenate(e4, 1);
    Eigen::Tensor<float_t, 4> d4 = this->dec_conv4.forward(d4_cat);
    d4 = this->dec_bn4.forward(d4);
    d4 = this->ac.ELU(d4);
    Eigen::Tensor<float_t, 4> d3_cat = d4.concatenate(e3, 1);
    Eigen::Tensor<float_t, 4> d3 = this->dec_conv3.forward(d3_cat);
    d3 = this->dec_bn3.forward(d3);
    d3 = this->ac.ELU(d3);
    Eigen::Tensor<float_t, 4> d2_cat = d3.concatenate(e2, 1);
    Eigen::Tensor<float_t, 4> d2 = this->dec_conv2.forward(d2_cat);
    d2 = this->dec_bn2.forward(d2);
    d2 = this->ac.ELU(d2);
    Eigen::Tensor<float_t, 4> d1_cat = d2.concatenate(e1, 1);
    Eigen::Tensor<float_t, 4> d1 = this->dec_conv1.forward(d1_cat);
    d1 = this->dec_bn1.forward(d1);
    d1 = this->ac.Softplus(d1);
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

Eigen::Tensor<float_t, 4> Model_CRN::viewBackward(Eigen::Tensor<float_t, 3> &input, Eigen::array<int64_t, 4> dims) {
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



