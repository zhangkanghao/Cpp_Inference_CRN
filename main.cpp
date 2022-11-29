#include "iostream"
#include "windows.h"
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

void sprint(Eigen::Tensor<float_t, 4> input) {
    const Eigen::Tensor<size_t, 4>::Dimensions &dim_inp = input.dimensions();
    cout << "Variable:" << endl;
    // 0 0
    cout << input(0, 0, 0, 0) << " " << input(0, 0, 0, 1) << " " << input(0, 0, 0, 2) << " ";
    cout << input(0, 0, 0, dim_inp[3] - 3) << " " << input(0, 0, 0, dim_inp[3] - 2) << " "
         << input(0, 0, 0, dim_inp[3] - 1);
    cout << endl;
    // 0 -1
    cout << input(0, 0, dim_inp[2] - 1, 0) << " " << input(0, 0, dim_inp[2] - 1, 1) << " "
         << input(0, 0, dim_inp[2] - 1, 2) << " ";
    cout << input(0, 0, dim_inp[2] - 1, dim_inp[3] - 3) << " " << input(0, 0, dim_inp[2] - 1, dim_inp[3] - 2) << " "
         << input(0, 0, dim_inp[2] - 1, dim_inp[3] - 1);
    cout << endl;
}

int main() {
    const char *wav_path = "C:/Users/65181/CLionProjects/Cpp_Inference_CRN/resources/SA2_destroyerops.wav";
    const char *out_path = "C:/Users/65181/CLionProjects/Cpp_Inference_CRN/resources/output_wav.wav";
    const char *path = "C:/Users/65181/CLionProjects/Cpp_Inference_CRN/resources/crn13.mat";
    Model_CRN model = Model_CRN(path);
    int64_t frame_size = 320, frame_shift = 160, fft_size = 320;
    Wav_File wav = Wav_File();
    wav.LoadWavFile(wav_path);
    float_t scale = wav.getNorm();

    // start timing
    DWORD star_time = GetTickCount();
    wav.setSTFT(frame_size, frame_shift, "sqrt");
    wav.newSTFT();
    wav.getMagnitude();
    wav.getPhase();

//    MATFile *pMatFile;
//    pMatFile = matOpen("C:/Users/65181/CLionProjects/Cpp_Inference_CRN/resources/inp_mag.mat", "r");
//    mxArray *pa = matGetVariable(pMatFile, "mag");
//    auto *values = (float_t *) mxGetData(pa);
//    long long dim1 = mxGetM(pa);
//    long long dim2 = mxGetN(pa);
//    Eigen::Tensor<float_t, 4> inpMag(1, 1, dim1, dim2);
//    Eigen::Tensor<float_t, 4> estMag(1, 1, dim1, dim2);
//
//    int idx = 0;
//    for (int i = 0; i < dim2; i++) {
//        for (int j = 0; j < dim1; j++) {
//            for (int k = 0; k < 1; k++) {
//                for (int l = 0; l < 1; l++) {
//                    inpMag(l, k, j, i) = values[idx++];
//                }
//            }
//        }
//    }
//    sprint(inpMag);
//
//    estMag = model.forward(inpMag);
//    pMatFile = matOpen("C:/Users/65181/CLionProjects/Cpp_Inference_CRN/resources/est_mag.mat", "w");
//    mxArray *pa2 = mxCreateDoubleMatrix(dim1, dim2, mxREAL);
//    idx = 0;
//    auto *pData1 = (double *) mxCalloc(dim1 * dim2, sizeof(double));
//    for (int i = 0; i < dim2; i++) {
//        for (int j = 0; j < dim1; j++) {
//            for (int k = 0; k < 1; k++) {
//                for (int l = 0; l < 1; l++) {
//                    pData1[idx++] = estMag(l, k, j, i);
//                }
//            }
//        }
//    }
//    mxSetData(pa2, pData1);
//    matPutVariable(pMatFile, "estmag", pa2);
//    matClose(pMatFile);
//    return 0;
//    for (int i = 0; i < dim1; i++) {
//        for (int j = 0; j < dim2; j++) {
//            inpMag(0, 0, i, j) = wav.spec_mag[i][j];
//        }
//    }
    Eigen::Tensor<float_t, 4> inpMag(1, 1, wav.frame_num, fft_size / 2 + 1);
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 1; j++) {
            for (int k = 0; k < wav.frame_num; k++) {
                for (int l = 0; l < fft_size / 2 + 1; l++) {
                    inpMag(i, j, k, l) = wav.spec_mag[k][l];
                }
            }
        }
    }
    Eigen::Tensor<float_t, 4> estMag;
    estMag = model.forward(inpMag);
    for (int i = 0; i < wav.frame_num; i++) {
        for (int j = 0; j < fft_size / 2 + 1; j++) {
            wav.spec_mag[i][j] = estMag(0, 0, i, j);
        }
    }
    wav.magToSpec();
    wav.newISTFT();
    DWORD end_time = GetTickCount();
    cout << "Program finished in ：" << (end_time - star_time) << "ms." << endl;
//     end timing
    wav.getNorm(scale);
    wav.WriteWavFile(out_path);
    wav.FreeSource();
    return 0;

}
