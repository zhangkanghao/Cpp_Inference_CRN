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

int main() {

    const char *path = "C:/Users/65181/CLionProjects/CRN/resources/crn.mat";
    const char *wav_path = "C:/Users/65181/CLionProjects/Cpp_Inference_CRN/resources/S006_ADTbabble_snr0_tgt.wav";
    const char *out_path = "C:/Users/65181/CLionProjects/Cpp_Inference_CRN/resources/output_wav.wav";
    Wav_File wav = Wav_File();
    wav.LoadWavFile(wav_path);
    wav.setSTFT(320, 160, "sqrt");
    wav.newSTFT();
    wav.newISTFT();
    wav.WriteWavFile(out_path);
    wav.FreeSource();
    return 0;

}
