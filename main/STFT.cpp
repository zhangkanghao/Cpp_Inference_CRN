//
// Created by Koer on 2022/11/24.
//

#include "../include/STFT.h"
#include "cmath"
#include "cstdio"
#include "cstdlib"
#include "memory.h"
#include "assert.h"

/************************************************************************
  fft(int n, float xRe[], float xIm[], float yRe[], float yIm[])
 ------------------------------------------------------------------------
  NOTE : This is copyrighted material, Not public domain. See below.
 ------------------------------------------------------------------------
  Input/output:
      int n          transformation length.
      float xRe[]   real part of input sequence.
      float xIm[]   imaginary part of input sequence.
      float yRe[]   real part of output sequence.
      float yIm[]   imaginary part of output sequence.
 ------------------------------------------------------------------------
  Function:
      The procedure performs a fast discrete Fourier transform (FFT) of
      a complex sequence, x, of an arbitrary length, n. The output, y,
      is also a complex sequence of length n.

      y[k] = sum(x[m]*exp(-i*2*pi*k*m/n), m=0..(n-1)), k=0,...,(n-1)

      The largest prime factor of n must be less than or equal to the
      constant maxPrimeFactor defined below.
 ------------------------------------------------------------------------
  Author:
      Jens Joergen Nielsen            For non-commercial use only.
      Bakkehusene 54                  A $100 fee must be paid if used
      DK-2970 Hoersholm               commercially. Please contact.
      DENMARK

      E-mail : jjn@get2net.dk   All rights reserved. October 2000.
      Homepage : http://home.get2net.dk/jjn
 ------------------------------------------------------------------------
  Implementation notes:
      The general idea is to factor the length of the DFT, n, into
      factors that are efficiently handled by the routines.

      A number of short DFT's are implemented with a minimum of
      arithmetical operations and using (almost) straight line code
      resulting in very fast execution when the factors of n belong
      to this set. Especially radix-10 is optimized.

      Prime factors, that are not in the set of short DFT's are handled
      with direct evaluation of the DFP expression.

      Please report any problems to the author.
      Suggestions and improvements are welcomed.
 ------------------------------------------------------------------------
  Benchmarks:
      The Microsoft Visual C++ compiler was used with the following
      compile options:
      /nologo /Gs /G2 /W4 /AH /Ox /D "NDEBUG" /D "_DOS" /FR
      and the FFTBENCH test executed on a 50MHz 486DX :

      Length  Time [s]  Accuracy [dB]

         128   0.0054     -314.8
         256   0.0116     -309.8
         512   0.0251     -290.8
        1024   0.0567     -313.6
        2048   0.1203     -306.4
        4096   0.2600     -291.8
        8192   0.5800     -305.1
         100   0.0040     -278.5
         200   0.0099     -280.3
         500   0.0256     -278.5
        1000   0.0540     -278.5
        2000   0.1294     -280.6
        5000   0.3300     -278.4
       10000   0.7133     -278.5
 ------------------------------------------------------------------------
  The following procedures are used :
      factorize       :  factor the transformation length.
      transTableSetup :  setup table with sofar-, actual-, and remainRadix.
      permute         :  permutation allows in-place calculations.
      twiddleTransf   :  twiddle multiplications and DFT's for one stage.
      initTrig        :  initialise sine/cosine table.
      fft_4           :  length 4 DFT, a la Nussbaumer.
      fft_5           :  length 5 DFT, a la Nussbaumer.
      fft_10          :  length 10 DFT using prime factor FFT.
      fft_odd         :  length n DFT, n odd.
*************************************************************************/

#define  maxPrimeFactor        37
#define  maxPrimeFactorDiv2    (maxPrimeFactor+1)/2
#define  maxFactorCount        20

static const float c3_1 = -1.5000000000000E+00f;  /*  c3_1 = cos(2*pi/3)-1;          */
static const float c3_2 = 8.6602540378444E-01f;  /*  c3_2 = sin(2*pi/3);            */

//static const float u5 = 1.2566370614359E+00;  /*  u5   = 2*pi/5;                 */
static const float c5_1 = -1.2500000000000E+00f;  /*  c5_1 = (cos(u5)+cos(2*u5))/2-1;*/
static const float c5_2 = 5.5901699437495E-01f;  /*  c5_2 = (cos(u5)-cos(2*u5))/2;  */
static const float c5_3 = -9.5105651629515E-01f;  /*  c5_3 = -sin(u5);               */
static const float c5_4 = -1.5388417685876E+00f;  /*  c5_4 = -(sin(u5)+sin(2*u5));   */
static const float c5_5 = 3.6327126400268E-01f;  /*  c5_5 = (sin(u5)-sin(2*u5));    */
static const float c8 = 7.0710678118655E-01f;  /*  c8 = 1/sqrt(2);    */

//static int isInited = 0; // 是否已经执行过transTableSetup

static float pi;

typedef struct FFTDataStruct {
    int groupOffset, dataOffset, adr;
    int groupNo, dataNo, blockNo, twNo;
    float omega, tw_re, tw_im;
    float twiddleRe[maxPrimeFactor], twiddleIm[maxPrimeFactor],
            trigRe[maxPrimeFactor], trigIm[maxPrimeFactor],
            zRe[maxPrimeFactor], zIm[maxPrimeFactor];
    float vRe[maxPrimeFactorDiv2], vIm[maxPrimeFactorDiv2];
    float wRe[maxPrimeFactorDiv2], wIm[maxPrimeFactorDiv2];
} FFTDataStruct;

void factorize(int n, int *nFact, int fact[]);
void transTableSetup(int sofar[], int actual[], int remain[],
                     int *nFact,
                     int *nPoints);
void permute(const int nPoint, const int nFact,
             const int fact[], const int remain[],
             float xRe[], float xIm[],
             float yRe[], float yIm[]);
void initTrig(FFTDataStruct *fftDataStruct, int radix);
void fft_4(float aRe[], float aIm[]);
void fft_5(float aRe[], float aIm[]);
void fft_8(FFTDataStruct *fftDataStruct);
void fft_10(FFTDataStruct *fftDataStruct);
void fft_odd(FFTDataStruct *fftDataStruct, int radix);
void twiddleTransf(FFTDataStruct *fftDataStruct, int sofarRadix, int radix, int remainRadix,
                   float yRe[], float yIm[]);
void initFFTDataStruct(FFTDataStruct *fftDataStruct);

void factorize(int n, int *nFact, int fact[]) {
    int i, j, k;
    int nRadix;
    int radices[7];
    int factors[maxFactorCount];
    memset(factors, 0, sizeof(factors));

    nRadix = 6;
    radices[1] = 2;
    radices[2] = 3;
    radices[3] = 4;
    radices[4] = 5;
    radices[5] = 8;
    radices[6] = 10;

    if (n == 1) {
        j = 1;
        factors[1] = 1;
    } else j = 0;
    i = nRadix;
    while ((n > 1) && (i > 0)) {
        if ((n % radices[i]) == 0) {
            n = n / radices[i];
            j = j + 1;
            factors[j] = radices[i];
        } else i = i - 1;
    }
    if (factors[j] == 2)   /*substitute factors 2*8 with 4*4 */
    {
        i = j - 1;
        while ((i > 0) && (factors[i] != 8)) i--;
        if (i > 0) {
            factors[j] = 4;
            factors[i] = 4;
        }
    }
    if (n > 1) {
        for (k = 2; k < sqrt(n) + 1; k++)
            while ((n % k) == 0) {
                n = n / k;
                j = j + 1;
                factors[j] = k;
            }
        if (n > 1) {
            j = j + 1;
            factors[j] = n;
        }
    }
    for (i = 1; i <= j; i++) {
        fact[i] = factors[j - i + 1];
    }
    *nFact = j;
}   /* factorize */

/****************************************************************************
  After N is factored the parameters that control the stages are generated.
  For each stage we have:
    sofar   : the product of the radices so far.
    actual  : the radix handled in this stage.
    remain  : the product of the remaining radices.
 ****************************************************************************/

void transTableSetup(int sofar[], int actual[], int remain[],
                     int *nFact,
                     int *nPoints) {
    int i;

    factorize(*nPoints, nFact, actual);
    if (actual[1] > maxPrimeFactor) {
        printf("\nPrime factor of FFT length too large : %6d", actual[1]);
        printf("\nPlease modify the value of maxPrimeFactor in mixfft.c");
        exit(1);
    }
    remain[0] = *nPoints;
    sofar[1] = 1;
    remain[1] = *nPoints / actual[1];
    for (i = 2; i <= *nFact; i++) {
        sofar[i] = sofar[i - 1] * actual[i - 1];
        remain[i] = remain[i - 1] / actual[i];
    }
}   /* transTableSetup */

/****************************************************************************
  The sequence y is the permuted input sequence x so that the following
  transformations can be performed in-place, and the final result is the
  normal order.
 ****************************************************************************/

void permute(const int nPoint, const int nFact,
             const int fact[], const int remain[],
             float xRe[], float xIm[],
             float yRe[], float yIm[]) {
    int i, j, k;
    int count[maxFactorCount];
    memset(count, 0, sizeof(count));

    for (i = 1; i <= nFact; i++) count[i] = 0;
    k = 0;
    for (i = 0; i <= nPoint - 2; i++) {
        yRe[i] = xRe[k];
        yIm[i] = xIm[k];
        j = 1;
        k = k + remain[j];
        count[1] = count[1] + 1;
        while (count[j] >= fact[j]) {
            count[j] = 0;
            k = k - remain[j - 1] + remain[j + 1];
            j = j + 1;
            count[j] = count[j] + 1;
        }
    }
    yRe[nPoint - 1] = xRe[nPoint - 1];
    yIm[nPoint - 1] = xIm[nPoint - 1];
}   /* permute */


/****************************************************************************
  Twiddle factor multiplications and transformations are performed on a
  group of data. The number of multiplications with 1 are reduced by skipping
  the twiddle multiplication of the first stage and of the first group of the
  following stages.
 ***************************************************************************/

void initTrig(FFTDataStruct *fftDataStruct, int radix) {
    int i;
    float w, xre, xim;

    w = 2 * pi / radix;
    fftDataStruct->trigRe[0] = 1;
    fftDataStruct->trigIm[0] = 0;
    xre = cosf(w);
    xim = -sinf(w);
    fftDataStruct->trigRe[1] = xre;
    fftDataStruct->trigIm[1] = xim;
    for (i = 2; i < radix; i++) {
        fftDataStruct->trigRe[i] = xre * fftDataStruct->trigRe[i - 1] - xim * fftDataStruct->trigIm[i - 1];
        fftDataStruct->trigIm[i] = xim * fftDataStruct->trigRe[i - 1] + xre * fftDataStruct->trigIm[i - 1];
    }
}   /* initTrig */

void fft_4(float aRe[], float aIm[]) {
    float t1_re, t1_im, t2_re, t2_im;
    float m2_re, m2_im, m3_re, m3_im;

    t1_re = aRe[0] + aRe[2];
    t1_im = aIm[0] + aIm[2];
    t2_re = aRe[1] + aRe[3];
    t2_im = aIm[1] + aIm[3];

    m2_re = aRe[0] - aRe[2];
    m2_im = aIm[0] - aIm[2];
    m3_re = aIm[1] - aIm[3];
    m3_im = aRe[3] - aRe[1];

    aRe[0] = t1_re + t2_re;
    aIm[0] = t1_im + t2_im;
    aRe[2] = t1_re - t2_re;
    aIm[2] = t1_im - t2_im;
    aRe[1] = m2_re + m3_re;
    aIm[1] = m2_im + m3_im;
    aRe[3] = m2_re - m3_re;
    aIm[3] = m2_im - m3_im;
}   /* fft_4 */


void fft_5(float aRe[], float aIm[]) {
    float t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
    float t4_re, t4_im, t5_re, t5_im;
    float m2_re, m2_im, m3_re, m3_im, m4_re, m4_im;
    float m1_re, m1_im, m5_re, m5_im;
    float s1_re, s1_im, s2_re, s2_im, s3_re, s3_im;
    float s4_re, s4_im, s5_re, s5_im;

    t1_re = aRe[1] + aRe[4];
    t1_im = aIm[1] + aIm[4];
    t2_re = aRe[2] + aRe[3];
    t2_im = aIm[2] + aIm[3];
    t3_re = aRe[1] - aRe[4];
    t3_im = aIm[1] - aIm[4];
    t4_re = aRe[3] - aRe[2];
    t4_im = aIm[3] - aIm[2];
    t5_re = t1_re + t2_re;
    t5_im = t1_im + t2_im;
    aRe[0] = aRe[0] + t5_re;
    aIm[0] = aIm[0] + t5_im;
    m1_re = c5_1 * t5_re;
    m1_im = c5_1 * t5_im;
    m2_re = c5_2 * (t1_re - t2_re);
    m2_im = c5_2 * (t1_im - t2_im);

    m3_re = -c5_3 * (t3_im + t4_im);
    m3_im = c5_3 * (t3_re + t4_re);
    m4_re = -c5_4 * t4_im;
    m4_im = c5_4 * t4_re;
    m5_re = -c5_5 * t3_im;
    m5_im = c5_5 * t3_re;

    s3_re = m3_re - m4_re;
    s3_im = m3_im - m4_im;
    s5_re = m3_re + m5_re;
    s5_im = m3_im + m5_im;
    s1_re = aRe[0] + m1_re;
    s1_im = aIm[0] + m1_im;
    s2_re = s1_re + m2_re;
    s2_im = s1_im + m2_im;
    s4_re = s1_re - m2_re;
    s4_im = s1_im - m2_im;

    aRe[1] = s2_re + s3_re;
    aIm[1] = s2_im + s3_im;
    aRe[2] = s4_re + s5_re;
    aIm[2] = s4_im + s5_im;
    aRe[3] = s4_re - s5_re;
    aIm[3] = s4_im - s5_im;
    aRe[4] = s2_re - s3_re;
    aIm[4] = s2_im - s3_im;
}   /* fft_5 */

void fft_8(FFTDataStruct *fftDataStruct) {
    float aRe[4], aIm[4], bRe[4], bIm[4], gem;

    aRe[0] = fftDataStruct->zRe[0];
    bRe[0] = fftDataStruct->zRe[1];
    aRe[1] = fftDataStruct->zRe[2];
    bRe[1] = fftDataStruct->zRe[3];
    aRe[2] = fftDataStruct->zRe[4];
    bRe[2] = fftDataStruct->zRe[5];
    aRe[3] = fftDataStruct->zRe[6];
    bRe[3] = fftDataStruct->zRe[7];

    aIm[0] = fftDataStruct->zIm[0];
    bIm[0] = fftDataStruct->zIm[1];
    aIm[1] = fftDataStruct->zIm[2];
    bIm[1] = fftDataStruct->zIm[3];
    aIm[2] = fftDataStruct->zIm[4];
    bIm[2] = fftDataStruct->zIm[5];
    aIm[3] = fftDataStruct->zIm[6];
    bIm[3] = fftDataStruct->zIm[7];

    fft_4(aRe, aIm);
    fft_4(bRe, bIm);

    gem = c8 * (bRe[1] + bIm[1]);
    bIm[1] = c8 * (bIm[1] - bRe[1]);
    bRe[1] = gem;
    gem = bIm[2];
    bIm[2] = -bRe[2];
    bRe[2] = gem;
    gem = c8 * (bIm[3] - bRe[3]);
    bIm[3] = -c8 * (bRe[3] + bIm[3]);
    bRe[3] = gem;

    fftDataStruct->zRe[0] = aRe[0] + bRe[0];
    fftDataStruct->zRe[4] = aRe[0] - bRe[0];
    fftDataStruct->zRe[1] = aRe[1] + bRe[1];
    fftDataStruct->zRe[5] = aRe[1] - bRe[1];
    fftDataStruct->zRe[2] = aRe[2] + bRe[2];
    fftDataStruct->zRe[6] = aRe[2] - bRe[2];
    fftDataStruct->zRe[3] = aRe[3] + bRe[3];
    fftDataStruct->zRe[7] = aRe[3] - bRe[3];

    fftDataStruct->zIm[0] = aIm[0] + bIm[0];
    fftDataStruct->zIm[4] = aIm[0] - bIm[0];
    fftDataStruct->zIm[1] = aIm[1] + bIm[1];
    fftDataStruct->zIm[5] = aIm[1] - bIm[1];
    fftDataStruct->zIm[2] = aIm[2] + bIm[2];
    fftDataStruct->zIm[6] = aIm[2] - bIm[2];
    fftDataStruct->zIm[3] = aIm[3] + bIm[3];
    fftDataStruct->zIm[7] = aIm[3] - bIm[3];
}   /* fft_8 */

void fft_10(FFTDataStruct *fftDataStruct) {
    float aRe[5], aIm[5], bRe[5], bIm[5];

    aRe[0] = fftDataStruct->zRe[0];
    bRe[0] = fftDataStruct->zRe[5];
    aRe[1] = fftDataStruct->zRe[2];
    bRe[1] = fftDataStruct->zRe[7];
    aRe[2] = fftDataStruct->zRe[4];
    bRe[2] = fftDataStruct->zRe[9];
    aRe[3] = fftDataStruct->zRe[6];
    bRe[3] = fftDataStruct->zRe[1];
    aRe[4] = fftDataStruct->zRe[8];
    bRe[4] = fftDataStruct->zRe[3];

    aIm[0] = fftDataStruct->zIm[0];
    bIm[0] = fftDataStruct->zIm[5];
    aIm[1] = fftDataStruct->zIm[2];
    bIm[1] = fftDataStruct->zIm[7];
    aIm[2] = fftDataStruct->zIm[4];
    bIm[2] = fftDataStruct->zIm[9];
    aIm[3] = fftDataStruct->zIm[6];
    bIm[3] = fftDataStruct->zIm[1];
    aIm[4] = fftDataStruct->zIm[8];
    bIm[4] = fftDataStruct->zIm[3];

    fft_5(aRe, aIm);
    fft_5(bRe, bIm);

    fftDataStruct->zRe[0] = aRe[0] + bRe[0];
    fftDataStruct->zRe[5] = aRe[0] - bRe[0];
    fftDataStruct->zRe[6] = aRe[1] + bRe[1];
    fftDataStruct->zRe[1] = aRe[1] - bRe[1];
    fftDataStruct->zRe[2] = aRe[2] + bRe[2];
    fftDataStruct->zRe[7] = aRe[2] - bRe[2];
    fftDataStruct->zRe[8] = aRe[3] + bRe[3];
    fftDataStruct->zRe[3] = aRe[3] - bRe[3];
    fftDataStruct->zRe[4] = aRe[4] + bRe[4];
    fftDataStruct->zRe[9] = aRe[4] - bRe[4];

    fftDataStruct->zIm[0] = aIm[0] + bIm[0];
    fftDataStruct->zIm[5] = aIm[0] - bIm[0];
    fftDataStruct->zIm[6] = aIm[1] + bIm[1];
    fftDataStruct->zIm[1] = aIm[1] - bIm[1];
    fftDataStruct->zIm[2] = aIm[2] + bIm[2];
    fftDataStruct->zIm[7] = aIm[2] - bIm[2];
    fftDataStruct->zIm[8] = aIm[3] + bIm[3];
    fftDataStruct->zIm[3] = aIm[3] - bIm[3];
    fftDataStruct->zIm[4] = aIm[4] + bIm[4];
    fftDataStruct->zIm[9] = aIm[4] - bIm[4];
}   /* fft_10 */

void fft_odd(FFTDataStruct *fftDataStruct, int radix) {
    float rere, reim, imre, imim;
    int i, j, k, n, max;

    n = radix;
    max = (n + 1) / 2;
    for (j = 1; j < max; j++) {
        fftDataStruct->vRe[j] = fftDataStruct->zRe[j] + fftDataStruct->zRe[n - j];
        fftDataStruct->vIm[j] = fftDataStruct->zIm[j] - fftDataStruct->zIm[n - j];
        fftDataStruct->wRe[j] = fftDataStruct->zRe[j] - fftDataStruct->zRe[n - j];
        fftDataStruct->wIm[j] = fftDataStruct->zIm[j] + fftDataStruct->zIm[n - j];
    }

    for (j = 1; j < max; j++) {
        fftDataStruct->zRe[j] = fftDataStruct->zRe[0];
        fftDataStruct->zIm[j] = fftDataStruct->zIm[0];
        fftDataStruct->zRe[n - j] = fftDataStruct->zRe[0];
        fftDataStruct->zIm[n - j] = fftDataStruct->zIm[0];
        k = j;
        for (i = 1; i < max; i++) {
            rere = fftDataStruct->trigRe[k] * fftDataStruct->vRe[i];
            imim = fftDataStruct->trigIm[k] * fftDataStruct->vIm[i];
            reim = fftDataStruct->trigRe[k] * fftDataStruct->wIm[i];
            imre = fftDataStruct->trigIm[k] * fftDataStruct->wRe[i];

            fftDataStruct->zRe[n - j] += rere + imim;
            fftDataStruct->zIm[n - j] += reim - imre;
            fftDataStruct->zRe[j] += rere - imim;
            fftDataStruct->zIm[j] += reim + imre;

            k = k + j;
            if (k >= n) k = k - n;
        }
    }
    for (j = 1; j < max; j++) {
        fftDataStruct->zRe[0] = fftDataStruct->zRe[0] + fftDataStruct->vRe[j];
        fftDataStruct->zIm[0] = fftDataStruct->zIm[0] + fftDataStruct->wIm[j];
    }
}   /* fft_odd */


void twiddleTransf(FFTDataStruct *fftDataStruct, int sofarRadix, int radix, int remainRadix,
                   float yRe[], float yIm[]) {   /* twiddleTransf */
    float cosw, sinw, gem;
    float t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
    float t4_re, t4_im, t5_re, t5_im;
    float m2_re, m2_im, m3_re, m3_im, m4_re, m4_im;
    float m1_re, m1_im, m5_re, m5_im;
    float s1_re, s1_im, s2_re, s2_im, s3_re, s3_im;
    float s4_re, s4_im, s5_re, s5_im;


    initTrig(fftDataStruct, radix);
    fftDataStruct->omega = 2 * pi / (float) (sofarRadix * radix);
    cosw = cosf(fftDataStruct->omega);
    sinw = -sinf(fftDataStruct->omega);
    fftDataStruct->tw_re = 1.0;
    fftDataStruct->tw_im = 0;
    fftDataStruct->dataOffset = 0;
    fftDataStruct->groupOffset = fftDataStruct->dataOffset;
    fftDataStruct->adr = fftDataStruct->groupOffset;
    for (fftDataStruct->dataNo = 0; fftDataStruct->dataNo < sofarRadix; fftDataStruct->dataNo++) {
        if (sofarRadix > 1) {
            fftDataStruct->twiddleRe[0] = 1.0;
            fftDataStruct->twiddleIm[0] = 0.0;
            fftDataStruct->twiddleRe[1] = fftDataStruct->tw_re;
            fftDataStruct->twiddleIm[1] = fftDataStruct->tw_im;
            for (fftDataStruct->twNo = 2; fftDataStruct->twNo < radix; fftDataStruct->twNo++) {
                fftDataStruct->twiddleRe[fftDataStruct->twNo] =
                        fftDataStruct->tw_re * fftDataStruct->twiddleRe[fftDataStruct->twNo - 1]
                        - fftDataStruct->tw_im * fftDataStruct->twiddleIm[fftDataStruct->twNo - 1];
                fftDataStruct->twiddleIm[fftDataStruct->twNo] =
                        fftDataStruct->tw_im * fftDataStruct->twiddleRe[fftDataStruct->twNo - 1]
                        + fftDataStruct->tw_re * fftDataStruct->twiddleIm[fftDataStruct->twNo - 1];
            }
            gem = cosw * fftDataStruct->tw_re - sinw * fftDataStruct->tw_im;
            fftDataStruct->tw_im = sinw * fftDataStruct->tw_re + cosw * fftDataStruct->tw_im;
            fftDataStruct->tw_re = gem;
        }
        for (fftDataStruct->groupNo = 0; fftDataStruct->groupNo < remainRadix; fftDataStruct->groupNo++) {
            if ((sofarRadix > 1) && (fftDataStruct->dataNo > 0)) {
                fftDataStruct->zRe[0] = yRe[fftDataStruct->adr];
                fftDataStruct->zIm[0] = yIm[fftDataStruct->adr];
                fftDataStruct->blockNo = 1;
                do {
                    fftDataStruct->adr = fftDataStruct->adr + sofarRadix;
                    fftDataStruct->zRe[fftDataStruct->blockNo] =
                            fftDataStruct->twiddleRe[fftDataStruct->blockNo] * yRe[fftDataStruct->adr]
                            - fftDataStruct->twiddleIm[fftDataStruct->blockNo] * yIm[fftDataStruct->adr];
                    fftDataStruct->zIm[fftDataStruct->blockNo] =
                            fftDataStruct->twiddleRe[fftDataStruct->blockNo] * yIm[fftDataStruct->adr]
                            + fftDataStruct->twiddleIm[fftDataStruct->blockNo] * yRe[fftDataStruct->adr];

                    fftDataStruct->blockNo++;
                } while (fftDataStruct->blockNo < radix);
            } else
                for (fftDataStruct->blockNo = 0; fftDataStruct->blockNo < radix; fftDataStruct->blockNo++) {
                    fftDataStruct->zRe[fftDataStruct->blockNo] = yRe[fftDataStruct->adr];
                    fftDataStruct->zIm[fftDataStruct->blockNo] = yIm[fftDataStruct->adr];
                    fftDataStruct->adr = fftDataStruct->adr + sofarRadix;
                }
            switch (radix) {
                case 2  :
                    gem = fftDataStruct->zRe[0] + fftDataStruct->zRe[1];
                    fftDataStruct->zRe[1] = fftDataStruct->zRe[0] - fftDataStruct->zRe[1];
                    fftDataStruct->zRe[0] = gem;
                    gem = fftDataStruct->zIm[0] + fftDataStruct->zIm[1];
                    fftDataStruct->zIm[1] = fftDataStruct->zIm[0] - fftDataStruct->zIm[1];
                    fftDataStruct->zIm[0] = gem;
                    break;
                case 3  :
                    t1_re = fftDataStruct->zRe[1] + fftDataStruct->zRe[2];
                    t1_im = fftDataStruct->zIm[1] + fftDataStruct->zIm[2];
                    fftDataStruct->zRe[0] = fftDataStruct->zRe[0] + t1_re;
                    fftDataStruct->zIm[0] = fftDataStruct->zIm[0] + t1_im;
                    m1_re = c3_1 * t1_re;
                    m1_im = c3_1 * t1_im;
                    m2_re = c3_2 * (fftDataStruct->zIm[1] - fftDataStruct->zIm[2]);
                    m2_im = c3_2 * (fftDataStruct->zRe[2] - fftDataStruct->zRe[1]);
                    s1_re = fftDataStruct->zRe[0] + m1_re;
                    s1_im = fftDataStruct->zIm[0] + m1_im;
                    fftDataStruct->zRe[1] = s1_re + m2_re;
                    fftDataStruct->zIm[1] = s1_im + m2_im;
                    fftDataStruct->zRe[2] = s1_re - m2_re;
                    fftDataStruct->zIm[2] = s1_im - m2_im;
                    break;
                case 4  :
                    t1_re = fftDataStruct->zRe[0] + fftDataStruct->zRe[2];
                    t1_im = fftDataStruct->zIm[0] + fftDataStruct->zIm[2];
                    t2_re = fftDataStruct->zRe[1] + fftDataStruct->zRe[3];
                    t2_im = fftDataStruct->zIm[1] + fftDataStruct->zIm[3];

                    m2_re = fftDataStruct->zRe[0] - fftDataStruct->zRe[2];
                    m2_im = fftDataStruct->zIm[0] - fftDataStruct->zIm[2];
                    m3_re = fftDataStruct->zIm[1] - fftDataStruct->zIm[3];
                    m3_im = fftDataStruct->zRe[3] - fftDataStruct->zRe[1];

                    fftDataStruct->zRe[0] = t1_re + t2_re;
                    fftDataStruct->zIm[0] = t1_im + t2_im;
                    fftDataStruct->zRe[2] = t1_re - t2_re;
                    fftDataStruct->zIm[2] = t1_im - t2_im;
                    fftDataStruct->zRe[1] = m2_re + m3_re;
                    fftDataStruct->zIm[1] = m2_im + m3_im;
                    fftDataStruct->zRe[3] = m2_re - m3_re;
                    fftDataStruct->zIm[3] = m2_im - m3_im;
                    break;
                case 5  :
                    t1_re = fftDataStruct->zRe[1] + fftDataStruct->zRe[4];
                    t1_im = fftDataStruct->zIm[1] + fftDataStruct->zIm[4];
                    t2_re = fftDataStruct->zRe[2] + fftDataStruct->zRe[3];
                    t2_im = fftDataStruct->zIm[2] + fftDataStruct->zIm[3];
                    t3_re = fftDataStruct->zRe[1] - fftDataStruct->zRe[4];
                    t3_im = fftDataStruct->zIm[1] - fftDataStruct->zIm[4];
                    t4_re = fftDataStruct->zRe[3] - fftDataStruct->zRe[2];
                    t4_im = fftDataStruct->zIm[3] - fftDataStruct->zIm[2];
                    t5_re = t1_re + t2_re;
                    t5_im = t1_im + t2_im;
                    fftDataStruct->zRe[0] = fftDataStruct->zRe[0] + t5_re;
                    fftDataStruct->zIm[0] = fftDataStruct->zIm[0] + t5_im;
                    m1_re = c5_1 * t5_re;
                    m1_im = c5_1 * t5_im;
                    m2_re = c5_2 * (t1_re - t2_re);
                    m2_im = c5_2 * (t1_im - t2_im);

                    m3_re = -c5_3 * (t3_im + t4_im);
                    m3_im = c5_3 * (t3_re + t4_re);
                    m4_re = -c5_4 * t4_im;
                    m4_im = c5_4 * t4_re;
                    m5_re = -c5_5 * t3_im;
                    m5_im = c5_5 * t3_re;

                    s3_re = m3_re - m4_re;
                    s3_im = m3_im - m4_im;
                    s5_re = m3_re + m5_re;
                    s5_im = m3_im + m5_im;
                    s1_re = fftDataStruct->zRe[0] + m1_re;
                    s1_im = fftDataStruct->zIm[0] + m1_im;
                    s2_re = s1_re + m2_re;
                    s2_im = s1_im + m2_im;
                    s4_re = s1_re - m2_re;
                    s4_im = s1_im - m2_im;

                    fftDataStruct->zRe[1] = s2_re + s3_re;
                    fftDataStruct->zIm[1] = s2_im + s3_im;
                    fftDataStruct->zRe[2] = s4_re + s5_re;
                    fftDataStruct->zIm[2] = s4_im + s5_im;
                    fftDataStruct->zRe[3] = s4_re - s5_re;
                    fftDataStruct->zIm[3] = s4_im - s5_im;
                    fftDataStruct->zRe[4] = s2_re - s3_re;
                    fftDataStruct->zIm[4] = s2_im - s3_im;
                    break;
                case 8  :
                    fft_8(fftDataStruct);
                    break;
                case 10  :
                    fft_10(fftDataStruct);
                    break;
                default  :
                    fft_odd(fftDataStruct, radix);
                    break;
            }
            fftDataStruct->adr = fftDataStruct->groupOffset;
            for (fftDataStruct->blockNo = 0; fftDataStruct->blockNo < radix; fftDataStruct->blockNo++) {
                yRe[fftDataStruct->adr] = fftDataStruct->zRe[fftDataStruct->blockNo];
                yIm[fftDataStruct->adr] = fftDataStruct->zIm[fftDataStruct->blockNo];
                fftDataStruct->adr = fftDataStruct->adr + sofarRadix;
            }
            fftDataStruct->groupOffset = fftDataStruct->groupOffset + sofarRadix * radix;
            fftDataStruct->adr = fftDataStruct->groupOffset;
        }
        fftDataStruct->dataOffset = fftDataStruct->dataOffset + 1;
        fftDataStruct->groupOffset = fftDataStruct->dataOffset;
        fftDataStruct->adr = fftDataStruct->groupOffset;
    }
}   /* twiddleTransf */

void initFFTDataStruct(FFTDataStruct *fftDataStruct) {
//    fftDataStruct->groupOffset = fftDataStruct->dataOffset = fftDataStruct->adr = 0;
//    fftDataStruct->groupNo = fftDataStruct->dataNo = fftDataStruct->blockNo = fftDataStruct->twNo = 0;
//    fftDataStruct->omega = fftDataStruct->tw_re = fftDataStruct->tw_im = 0.0f;
//    memset(fftDataStruct->twiddleRe, 0, maxPrimeFactor * sizeof(float));
//    memset(fftDataStruct->twiddleIm, 0, maxPrimeFactor* sizeof(float));
//    memset(fftDataStruct->trigRe, 0, maxPrimeFactor* sizeof(float));
//    memset(fftDataStruct->trigIm, 0, maxPrimeFactor* sizeof(float));
//    memset(fftDataStruct->zRe, 0, maxPrimeFactor* sizeof(float));
//    memset(fftDataStruct->zIm, 0, maxPrimeFactor* sizeof(float));
//    memset(fftDataStruct->vRe, 0, maxPrimeFactorDiv2* sizeof(float));
//    memset(fftDataStruct->vIm, 0, maxPrimeFactorDiv2* sizeof(float));
//    memset(fftDataStruct->wRe, 0, maxPrimeFactorDiv2* sizeof(float));
//    memset(fftDataStruct->wIm, 0, maxPrimeFactorDiv2* sizeof(float));
    memset(fftDataStruct, 0, sizeof(FFTDataStruct));

}

struct fft_struct {
    int sofarRadix[maxFactorCount];
    int actualRadix[maxFactorCount];
    int remainRadix[maxFactorCount];
    int nFactor;
};

static const struct fft_struct nb_fft = {{0, 1, 4, 16},{0, 4, 4, 10},{160, 40, 10, 1}, 3};
static const struct fft_struct wb_fft = {{0, 1, 4, 32},{0, 4, 8, 10},{320, 80, 10, 1}, 3};
static const struct fft_struct wb_fft_16ms = {{0, 1, 8, 64},{0, 8, 8, 8},{512, 64, 8, 1}, 3};
static const struct fft_struct swb_fft = {{0, 1, 8, 64},{0, 8, 8, 10},{640, 80, 10, 1}, 3};
static const struct fft_struct swb_fft_16ms = {{0,1,4,16,128}, {0,4,4,8,8},{1024,256,64,8,1}, 4};
static const struct fft_struct fb_fft = {{0, 1, 3, 12, 96},{0, 3, 4, 8, 10},{960, 320, 80, 10, 1}, 4};
static const struct fft_struct fb_fft_16ms = {{0, 1, 3, 24, 192},{0, 3, 8, 8, 8}, {1536, 512, 64, 8, 1}, 4};

void fft(int n, float xRe[], float xIm[],
         float yRe[], float yIm[]) {
#if 1
    FFTDataStruct fftDataStruct;
    initFFTDataStruct(&fftDataStruct);

    const struct fft_struct *cur = NULL;

    switch (n) {
        case 160:
            cur = &nb_fft;
            break;
        case 320:
            cur = &wb_fft;
            break;
        case 512: // 16ms 16k
            cur = &wb_fft_16ms;
            break;
        case 640:
            cur = &swb_fft;
            break;
        case 1024:
            cur = &swb_fft_16ms;
            break;
        case 960:
            cur = &fb_fft;
            break;
        case 1536:
            cur = &fb_fft_16ms;
            break;
        default:
            assert(0 && "not support this fft point");
            break;
    }

    int count;

    pi = 3.141592653f;

    // 动态调整参数表
    // int nFactor = 3;
    // transTableSetup(sofarRadix, actualRadix, remainRadix, &nFactor, &n);
    // permute(n, nFactor, actualRadix, remainRadix, xRe, xIm, yRe, yIm);
    permute(n, cur->nFactor, cur->actualRadix, cur->remainRadix, xRe, xIm, yRe, yIm);

    for (count = 1; count <= cur->nFactor; count++) {
        twiddleTransf(&fftDataStruct, cur->sofarRadix[count], cur->actualRadix[count], cur->remainRadix[count],
                      yRe, yIm);
    }
#else
    FFTDataStruct fftDataStruct;
    initFFTDataStruct(&fftDataStruct);
    int sofarRadix[maxFactorCount] = {0, 1, 4, 32};
    int actualRadix[maxFactorCount] = {0, 4, 8, 10};
    int remainRadix[maxFactorCount] = {320, 80, 10, 1};
    int nFactor = 3;
    int count;
    pi = 3.141592653f;
    transTableSetup(sofarRadix, actualRadix, remainRadix, &nFactor, &n);
    permute(n, nFactor, actualRadix, remainRadix, xRe, xIm, yRe, yIm);
    for (count = 1; count <= nFactor; count++)
        twiddleTransf(&fftDataStruct, sofarRadix[count], actualRadix[count], remainRadix[count],
                      yRe, yIm);
#endif

}   /* fft */