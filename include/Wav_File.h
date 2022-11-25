#ifndef CRN_WAV_FILE_H
#define CRN_WAV_FILE_H

#include "Eigen"
#include "STFT.h"

#define PI 3.141592653589793f
struct Complex {
    float_t real, imag;
};

class Wav_File {
public:
    int16_t *ip_raw;
    float_t *fp_raw;
    float_t *data;
    size_t wav_size;
    Complex **spec;
    float **spec_real;
    float **spec_imag;
    float_t **mag;
    float_t **phase;


    /* 无参数构造函数 */
    Wav_File();

    /* 加载语音文件 */
    void LoadWavFile(const char *file_path);

    /* 写出语音文件 */
    void WriteWavFile(const char *dest_path);

    /* 加载语音文件 */
    void LoadPcmFile(const char *file_path);

    /* 写出语音文件 */
    void WritePcmFile(const char *dest_path);

    /* 初始化STFT */
    void setSTFT(int16_t frame_size, int16_t frame_shift, const char *window);

    /* 设置窗函数 */
    void setWindow(const char *winType);

    /* 计算STFT */
    void STFT();

    void newSTFT();

    /* 计算逆STFT */
    void ISTFT();

    void newISTFT();

    /* 释放内存 */
    void FreeSource();

    /* 计算幅度谱 */
    void getMagnitude();

    /* 计算相位谱 */
    void getPhase();

    /* 幅度相位计算频谱 */
    void magToSpec();

private:
    /* wav info */
    char id_riff[5], id_wave[5], id_fmt[5], id_data[5];
    int32_t file_size;
    int16_t format_tag, channels, block_align, bits_per_sample, cb_size;
    int32_t format_length, sample_rate, avg_bytes_sec, data_size;

    const char *win_type;
    int16_t frame_size;
    int16_t frame_shift;
    int16_t frame_num;
    int16_t fft_size;
    float_t *window;
    float_t *real;
    float_t *imag;

    void FFT(float_t param_real[], float_t param_imag[], int16_t param_n);

    void bitrp(float_t param_real[], float_t param_imag[], int16_t param_n);

    void IFFT(float_t param_real[], float_t param_imag[], int16_t param_n);

};


#endif //CRN_WAV_FILE_H