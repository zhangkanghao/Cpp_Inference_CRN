//
// Created by 65181 on 2022/11/1.
//

#include "../include/Wav_File.h"
#include "Eigen"
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <iostream>


Wav_File::Wav_File() {
    this->ip_raw = nullptr;
    this->fp_raw = nullptr;
    this->wav_size = 0;
}

void Wav_File::LoadWavFile(const char *file_path) {
    FILE *fp = fopen(file_path, "rb");
    if (fp) {
        fread(this->id_riff, sizeof(char), 4, fp);
        this->id_riff[4] = '\0';
        fread(&this->file_size, sizeof(int16_t), 2, fp);
        fread(this->id_wave, sizeof(char), 4, fp);
        this->id_wave[4] = '\0';
        fread(this->id_fmt, sizeof(char), 4, fp);
        this->id_fmt[4] = '\0';
        fread(&this->format_length, sizeof(int16_t), 2, fp);
        fread(&this->format_tag, sizeof(int16_t), 1, fp);
        fread(&this->channels, sizeof(int16_t), 1, fp);
        fread(&this->sample_rate, sizeof(int16_t), 2, fp);
        fread(&this->avg_bytes_sec, sizeof(int16_t), 2, fp);
        fread(&this->block_align, sizeof(int16_t), 1, fp);
        fread(&this->bits_per_sample, sizeof(int16_t), 1, fp);
        if (this->format_tag == 1) {
            fread(this->id_data, sizeof(char), 4, fp);
            this->id_data[4] = '\0';
            fread(&this->data_size, sizeof(int16_t), 2, fp);
            this->wav_size = this->data_size / sizeof(int16_t);
            this->ip_raw = (int16_t *) malloc(this->data_size);
            fread(this->ip_raw, sizeof(int16_t), this->wav_size, fp);

            this->data = (float_t *) malloc(this->wav_size * sizeof(float_t));
            for (int i = 0; i < this->wav_size; i++) {
                this->data[i] = (float_t) (this->ip_raw[i] * 1.0 / 32768);
            }

        } else if (this->format_tag == 3) {
            fread(&this->cb_size, sizeof(int16_t), 1, fp);
            fread(this->id_data, sizeof(char), 4, fp);
            this->id_data[4] = '\0';
            fread(&this->data_size, sizeof(int16_t), 2, fp);
            this->wav_size = this->data_size / sizeof(float);
            this->fp_raw = (float *) malloc(this->data_size);
            fread(this->fp_raw, sizeof(float), this->wav_size, fp);
            this->data = (float_t *) malloc(this->wav_size * sizeof(float_t));
            for (int i = 0; i < this->wav_size; i++) {
                this->data[i] = this->fp_raw[i];
            }
        }
    }
    fclose(fp);

}

void Wav_File::WriteWavFile(const char *dest_path) {
    for (int i = 0; i < wav_size; i++) {
        if (format_tag == 1)
            this->ip_raw[i] = (int16_t) (this->data[i] * 32768);
        else
            this->fp_raw[i] = this->data[i];
    }

    FILE *fp = fopen(dest_path, "wb");
    if (fp) {

        fwrite(this->id_riff, sizeof(char), 4, fp);
        fwrite(&this->file_size, sizeof(int16_t), 2, fp);
        fwrite(this->id_wave, sizeof(char), 4, fp);
        fwrite(this->id_fmt, sizeof(char), 4, fp);
        fwrite(&this->format_length, sizeof(int16_t), 2, fp);
        fwrite(&this->format_tag, sizeof(int16_t), 1, fp);
        fwrite(&this->channels, sizeof(int16_t), 1, fp);
        fwrite(&this->sample_rate, sizeof(int16_t), 2, fp);
        fwrite(&this->avg_bytes_sec, sizeof(int16_t), 2, fp);
        fwrite(&this->block_align, sizeof(int16_t), 1, fp);
        fwrite(&this->bits_per_sample, sizeof(int16_t), 1, fp);
        if (this->format_tag == 3) {
            fwrite(&this->cb_size, sizeof(int16_t), 1, fp);
        }
        fwrite(this->id_data, sizeof(char), 4, fp);
        fwrite(&this->data_size, sizeof(int16_t), 2, fp);
        if (this->format_tag == 1) {
            fwrite(this->ip_raw, sizeof(int16_t), this->wav_size, fp);
        } else if (this->format_tag == 3) {
            fwrite(this->fp_raw, sizeof(float), this->wav_size, fp);
        }

    }
    fclose(fp);

}

void Wav_File::LoadPcmFile(const char *file_path) {
    FILE *fp = fopen(file_path, "rb");

    // 计算文件长度
    fseek(fp, 0L, SEEK_END);
    this->wav_size = ftell(fp) / sizeof(float);
    fseek(fp, 0L, SEEK_SET);

    //申请内存空间
    this->fp_raw = (float *) malloc(this->wav_size * sizeof(float));
    fread(this->fp_raw, sizeof(float), this->wav_size, fp);
    fclose(fp);

    for (int i = 0; i < this->wav_size; i++) {
        this->data[i] = this->fp_raw[i];
    }
}

void Wav_File::WritePcmFile(const char *dest_path) {
    for (int i = 0; i < this->wav_size; i++) {
        this->fp_raw[i] = this->data[i];
    }
    FILE *fp = fopen(dest_path, "wb");
    fwrite(this->fp_raw, sizeof(float), this->wav_size, fp);
    fclose(fp);
}

void Wav_File::setSTFT(int64_t frame_size, int64_t frame_shift, const char *win_type) {
    this->frame_size = frame_size;
    this->frame_shift = frame_shift;
    this->fft_size = this->frame_size;
    this->frame_num = this->wav_size / frame_shift + 1;

    this->win_type = win_type;
    this->setWindow(this->win_type);

    this->spec = (Complex **) malloc((this->fft_size / 2 + 1) * sizeof(*spec));
    for (int i = 0; i < (this->fft_size / 2 + 1); i++)
        this->spec[i] = (Complex *) malloc(this->frame_num * sizeof(Complex));

    this->mag = (float_t **) malloc((this->fft_size / 2 + 1) * sizeof(*this->mag));
    for (int i = 0; i < (this->fft_size / 2 + 1); i++)
        this->mag[i] = (float_t *) malloc(frame_num * sizeof(float_t));

    this->phase = (float_t **) malloc((this->fft_size / 2 + 1) * sizeof(*this->phase));
    for (int i = 0; i < (this->fft_size / 2 + 1); i++)
        this->phase[i] = (float_t *) malloc(frame_num * sizeof(float_t));

    // nfft
    this->spec_real = (float_t **) malloc(this->frame_num * sizeof(*spec_real));
    for (int i = 0; i < this->frame_num; i++)
        this->spec_real[i] = (float_t *) malloc((this->fft_size / 2 + 1) * sizeof(float_t));

    this->spec_imag = (float_t **) malloc(this->frame_num * sizeof(*spec_imag));
    for (int i = 0; i < this->frame_num; i++)
        this->spec_imag[i] = (float_t *) malloc((this->fft_size / 2 + 1) * sizeof(float_t));

    this->spec_mag = (float_t **) malloc(this->frame_num * sizeof(*spec_mag));
    for (int i = 0; i < this->frame_num; i++)
        this->spec_mag[i] = (float_t *) malloc((this->fft_size / 2 + 1) * sizeof(float_t));

    this->spec_pha = (float_t **) malloc(this->frame_num * sizeof(*spec_pha));
    for (int i = 0; i < this->frame_num; i++)
        this->spec_pha[i] = (float_t *) malloc((this->fft_size / 2 + 1) * sizeof(float_t));


}

void Wav_File::setWindow(const char *winType) {
    double_t a0, a1;
    this->window = (float_t *) malloc(this->frame_size * sizeof(float_t));
    int frac = this->frame_size - 1;
    if (strcmp(winType, "hamming") == 0) {
        for (int i = 0; i < this->frame_size; i++) {
            this->window[i] = float(0.540000000 - 0.460000000 * cos(PI * 2 * i / frac));
        }
    } else if (strcmp(winType, "hanning") == 0) {
        for (int i = 0; i < this->frame_size; i++) {
            this->window[i] = float(0.500000000 - 0.500000000 * cos(PI * 2 * i / frac));
        }
    } else if (strcmp(winType, "sqrt") == 0) {
        for (int i = 0; i < this->frame_size; i++) {
            this->window[i] = (float) sqrtf((0.5f - 0.5f * cosf(i / (float) (this->frame_size - 1) * 2 * PI)));
        }
    }


}

void Wav_File::STFT() {
    std::vector<float_t> padding_data;
    for (int i = 0; i < this->frame_shift; i++)
        padding_data.push_back(this->data[this->frame_shift - i]);
    for (int i = 0; i < this->wav_size; i++)
        padding_data.push_back(this->data[i]);
    for (int i = 0; i < this->frame_shift; i++)
        padding_data.push_back(this->data[this->wav_size - i - 2]);

    this->real = (float_t *) malloc(sizeof(float_t) * this->frame_size);
    this->imag = (float_t *) malloc(sizeof(float_t) * this->frame_size);

    for (int i = 0; i < this->frame_num; i++) {
        for (int j = 0; j < this->frame_size; j++) {
            this->real[j] = padding_data[i * this->frame_shift + j] * this->window[j];
            this->imag[j] = 0.0;
        }

        FFT(this->real, this->imag, this->fft_size);

        for (int j = 0; j <= this->fft_size / 2; j++) {
            this->spec[j][i].real = this->real[j];
            this->spec[j][i].imag = this->imag[j];
        }
    }
}

void Wav_File::ISTFT() {
    this->real = (float_t *) malloc(sizeof(float_t) * this->frame_size);
    this->imag = (float_t *) malloc(sizeof(float_t) * this->frame_size);
    auto *temp = (float_t *) malloc(sizeof(float_t) * (data_size + 2 * frame_shift));
    auto *ola = (float_t *) malloc(sizeof(float_t) * (data_size + 2 * frame_shift));

    for (int i = 0; i < this->frame_num; i++) {
        //��N_FFT/2+1����N_FFT(����Գ�)
        for (int j = 0; j < this->fft_size / 2 + 1; j++) {
            real[j] = spec[j][i].real;
            imag[j] = spec[j][i].imag;
        }
        for (int j = this->fft_size / 2 + 1; j < this->fft_size; j++) {
            real[j] = spec[this->fft_size - j][i].real;
            imag[j] = -spec[this->fft_size - j][i].imag;
        }

        IFFT(real, imag, this->fft_size);

        for (int j = 0; j < frame_size; j++) {
            temp[i * frame_shift + j] += (real[j] * this->window[j]);
            ola[i * frame_shift + j] += (this->window[j] * this->window[j]);
        }
    }

    for (int i = 0; i < (frame_num + 1) * frame_shift; i++) {
        if (temp[i] != 0) {
            temp[i] /= ola[i];
        }
    }
    for (int i = 0; i < (frame_num - 1) * frame_shift; i++)
        this->data[i] = temp[i + frame_shift];

//    超出部分就保持原样,不降噪就好了
//    int sub = this->wav_size - (frame_num - 1) * frame_shift;
//    this->file_size -= sub;
//    this->wav_size -= sub;
//    this->data_size -= sub;
}

void Wav_File::FFT(float_t param_real[], float_t param_imag[], int16_t param_n) {
    float_t treal, timag, ureal, uimag, arg;
    int32_t m, k, j, t, index1, index2;
    auto *wreal = (float_t *) malloc(frame_shift * sizeof(float_t));
    auto *wimag = (float_t *) malloc(frame_shift * sizeof(float_t));


    this->bitrp(param_real, param_imag, param_n);

    arg = -2 * PI / param_n;
    treal = cosf(arg);
    timag = sinf(arg);
    wreal[0] = 1.0;
    wimag[0] = 0.0;
    for (j = 1; j < param_n / 2; j++) {
        wreal[j] = wreal[j - 1] * treal - wimag[j - 1] * timag;
        wimag[j] = wreal[j - 1] * timag + wimag[j - 1] * treal;
    }

    for (m = 2; m <= param_n; m *= 2) {
        for (k = 0; k < param_n; k += m) {
            for (j = 0; j < m / 2; j++) {
                index1 = k + j;
                index2 = index1 + m / 2;
                t = param_n * j / m;
                treal = wreal[t] * param_real[index2] - wimag[t] * param_imag[index2];
                timag = wreal[t] * param_imag[index2] + wimag[t] * param_real[index2];
                ureal = param_real[index1];
                uimag = param_imag[index1];
                param_real[index1] = ureal + treal;
                param_imag[index1] = uimag + timag;
                param_real[index2] = ureal - treal;
                param_imag[index2] = uimag - timag;
            }
        }
    }
    free(wreal);
    free(wimag);
}

void Wav_File::IFFT(float_t param_real[], float_t param_imag[], int16_t param_n) {
    float_t treal, timag, ureal, uimag, arg;
    int m, k, j, t, index1, index2;
    auto *wreal = (float_t *) malloc(frame_shift * sizeof(float_t));
    auto *wimag = (float_t *) malloc(frame_shift * sizeof(float_t));

    bitrp(param_real, imag, param_n);

    arg = 2 * PI / param_n;
    treal = cosf(arg);
    timag = sinf(arg);
    wreal[0] = 1.0;
    wimag[0] = 0.0;
    for (j = 1; j < param_n / 2; j++) {
        wreal[j] = wreal[j - 1] * treal - wimag[j - 1] * timag;
        wimag[j] = wreal[j - 1] * timag + wimag[j - 1] * treal;
    }

    for (m = 2; m <= param_n; m *= 2) {
        for (k = 0; k < param_n; k += m) {
            for (j = 0; j < m / 2; j++) {
                index1 = k + j;
                index2 = index1 + m / 2;
                t = param_n * j / m;
                treal = wreal[t] * param_real[index2] - wimag[t] * param_imag[index2];
                timag = wreal[t] * param_imag[index2] + wimag[t] * param_real[index2];
                ureal = param_real[index1];
                uimag = param_imag[index1];
                param_real[index1] = ureal + treal;
                param_imag[index1] = uimag + timag;
                param_real[index2] = ureal - treal;
                param_imag[index2] = uimag - timag;
            }
        }
    }

    for (j = 0; j < param_n; j++) {
        real[j] /= param_n;
        imag[j] /= param_n;
    }
    free(wreal);
    free(wimag);
}

void Wav_File::bitrp(float_t param_real[], float_t param_imag[], int16_t param_n) {
    int i, j, a, b, p;

    for (i = 1, p = 0; i < param_n; i *= 2) {
        p++;
    }
    for (i = 0; i < param_n; i++) {
        a = i;
        b = 0;
        for (j = 0; j < p; j++) {
            b = (b << 1) + (a & 1);    // b = b * 2 + a % 2;
            a >>= 1;        // a = a / 2;
        }
        if (b > i) {
            std::swap(param_real[i], param_real[b]);
            std::swap(param_imag[i], param_imag[b]);
        }
    }
}

void Wav_File::newSTFT() {
    std::vector<float_t> pad_data;
    for (int i = 0; i < this->frame_shift; i++)
        pad_data.push_back(0.0);
    for (int i = 0; i < this->wav_size; i++)
        pad_data.push_back(this->data[i]);
    for (int i = 0; i < this->frame_shift; i++)
        pad_data.push_back(0.0);

    auto fft_r = (float_t *) malloc(sizeof(float_t) * this->frame_size);
    auto fft_i = (float_t *) malloc(sizeof(float_t) * this->frame_size);
    auto res_r = (float_t *) malloc(sizeof(float_t) * this->frame_size);
    auto res_i = (float_t *) malloc(sizeof(float_t) * this->frame_size);

    for (int i = 0; i < this->frame_num; i++) {
        for (int j = 0; j < this->frame_size; j++) {
            fft_r[j] = pad_data[i * this->frame_shift + j] * this->window[j];
            fft_i[j] = 0.0;
        }

        fft(this->fft_size, fft_r, fft_i, res_r, res_i);

        for (int j = 0; j < this->fft_size / 2 + 1; j++) {
            this->spec_real[i][j] = res_r[j];
            this->spec_imag[i][j] = res_i[j];
        }
    }
}

void Wav_File::newISTFT() {
    auto fft_r = (float_t *) malloc(sizeof(float_t) * this->frame_size);
    auto fft_i = (float_t *) malloc(sizeof(float_t) * this->frame_size);
    auto res_r = (float_t *) malloc(sizeof(float_t) * this->frame_size);
    auto res_i = (float_t *) malloc(sizeof(float_t) * this->frame_size);

    auto *temp = (float_t *) malloc(sizeof(float_t) * (this->wav_size + 2 * this->frame_shift));
    memset(temp, 0, sizeof(float_t) * (this->wav_size + 2 * this->frame_shift));
    for (int i = 0; i < this->frame_num; i++) {
        for (int j = 0; j < this->fft_size / 2 + 1; j++) {
            fft_r[j] = this->spec_real[i][j] / this->frame_size;
            fft_i[j] = -this->spec_imag[i][j] / this->frame_size;
        }
        for (int j = this->fft_size / 2 + 1; j < this->fft_size; j++) {
            fft_r[j] = fft_r[this->fft_size - j];
            fft_i[j] = -fft_i[this->fft_size - j];
        }

        fft(this->fft_size, fft_r, fft_i, res_r, res_i);

        for (int j = 0; j < this->frame_size; j++) {
            temp[i * this->frame_shift + j] += (res_r[j] * this->window[j]);
        }
    }

    for (int i = 0; i < wav_size; i++) {
        this->data[i] = temp[i + this->frame_shift];
    }
}

void Wav_File::FreeSource() {
    free(this->ip_raw);
    free(this->fp_raw);
    free(this->spec);
    free(this->mag);
    free(this->phase);
    free(this->spec_real);
    free(this->spec_imag);
    free(this->spec_mag);
    free(this->spec_pha);

}

void Wav_File::getMagnitude() {
    for (int i = 0; i < this->frame_num; i++) {
        for (int j = 0; j < this->fft_size / 2 + 1; j++) {
            this->spec_mag[i][j] = sqrtf(powf(spec_real[i][j], 2) + powf(spec_imag[i][j], 2));
        }
    }
}

void Wav_File::getPhase() {
    for (int i = 0; i < this->frame_num; i++) {
        for (int j = 0; j < this->fft_size / 2 + 1; j++) {
            this->spec_pha[i][j] = atan2f(spec_imag[i][j], spec_real[i][j]);
        }
    }
}

void Wav_File::magToSpec() {
    for (int i = 0; i < this->frame_num; i++) {
        for (int j = 0; j < this->fft_size / 2 + 1; j++) {
            spec_real[i][j] = this->spec_mag[i][j] * cosf(this->spec_pha[i][j]);
            spec_imag[i][j] = this->spec_mag[i][j] * sinf(this->spec_pha[i][j]);
        }
    }
}

float_t Wav_File::getNorm(float_t scale) {
    if (scale == 0) {
        float_t square = 0.0;
        for (int i = 0; i < this->wav_size; i++) {
            square += powf(this->data[i], 2);
        }
        scale = sqrtf((float_t) this->wav_size / square);
        for (int i = 0; i < this->wav_size; i++) {
            this->data[i] *= scale;
        }
    } else {
        for (int i = 0; i < this->wav_size; i++) {
            this->data[i] /= scale;
        }
    }
    return scale;
}
