// test_norm_ffn.c
#include "tensor.h"
#include "tensor_ops.h"
#include "norm.h"
#include "ffn.h"
#include <stdio.h>
#include <math.h>

#define EPS_CHECK 1e-6

int main(void) {
    printf("=== Test 1: RMSNorm (hitung manual vs implementasi) ===\n");
    int shape[] = {1, 2};
    Tensor *x = tensor_create(shape, 2, 0);
    Tensor *gain = tensor_create((int[]){2}, 1, 0);
    Tensor *out = tensor_create(shape, 2, 0);

    x->data[0] = 3.0;
    x->data[1] = 4.0;
    gain->data[0] = 1.0;
    gain->data[1] = 1.0;

    double eps = 1e-6;
    int rc = rmsnorm_forward(x, gain, out, eps);
    printf("rmsnorm_forward return: %d (harusnya 0)\n", rc);

    double expected_rms = sqrt(12.5 + eps);
    double expected_0 = 3.0 / expected_rms;
    double expected_1 = 4.0 / expected_rms;

    printf("out[0] = %.6f (harusnya %.6f)\n", out->data[0], expected_0);
    printf("out[1] = %.6f (harusnya %.6f)\n", out->data[1], expected_1);

    int rmsnorm_pass = (fabs(out->data[0] - expected_0) < EPS_CHECK) &&
                        (fabs(out->data[1] - expected_1) < EPS_CHECK);
    printf("RMSNorm: %s\n\n", rmsnorm_pass ? "LOLOS" : "GAGAL");

    printf("=== Test 2: SiLU activation (nilai known) ===\n");
    double silu_0 = silu(0.0);
    double silu_1 = silu(1.0);
    double silu_neg1 = silu(-1.0);

    printf("silu(0)  = %.6f (harusnya 0.000000)\n", silu_0);
    printf("silu(1)  = %.6f (harusnya 0.731059)\n", silu_1);
    printf("silu(-1) = %.6f (harusnya -0.268941)\n", silu_neg1);

    int silu_pass = (fabs(silu_0 - 0.0) < EPS_CHECK) &&
                     (fabs(silu_1 - 0.731059) < 1e-5) &&
                     (fabs(silu_neg1 - (-0.268941)) < 1e-5);
    printf("SiLU: %s\n\n", silu_pass ? "LOLOS" : "GAGAL");

    printf("=== Test 3: SwiGLU FFN (shape check + konsistensi) ===\n");
    int seq_len = 3, dim = 4, hidden = 6;
    Tensor *xf = tensor_create((int[]){seq_len, dim}, 2, 0);
    Tensor *W1 = tensor_create((int[]){dim, hidden}, 2, 0);
    Tensor *W3 = tensor_create((int[]){dim, hidden}, 2, 0);
    Tensor *W2 = tensor_create((int[]){hidden, dim}, 2, 0);
    Tensor *outf = tensor_create((int[]){seq_len, dim}, 2, 0);
    Tensor *scratch1 = tensor_create((int[]){seq_len, hidden}, 2, 0);
    Tensor *scratch2 = tensor_create((int[]){seq_len, hidden}, 2, 0);

    tensor_fill_random(xf, -1.0, 1.0, 1);
    tensor_fill_random(W1, -0.5, 0.5, 2);
    tensor_fill_random(W3, -0.5, 0.5, 3);
    tensor_fill_random(W2, -0.5, 0.5, 4);

    int rc2 = swiglu_forward(xf, W1, W3, W2, outf, scratch1, scratch2);
    printf("swiglu_forward return: %d (harusnya 0)\n", rc2);
    tensor_print(outf, "Output SwiGLU");

    printf("\n=== Test 4: Shape salah (harus gagal) ===\n");
    Tensor *bad_out = tensor_create((int[]){seq_len, dim + 1}, 2, 0);
    int fail_rc = swiglu_forward(xf, W1, W3, W2, bad_out, scratch1, scratch2);
    printf("swiglu_forward dengan shape salah: %d (harusnya -1)\n", fail_rc);

    int all_pass = rmsnorm_pass && silu_pass && (rc == 0) && (rc2 == 0) && (fail_rc == -1);

    printf("\n=== HASIL AKHIR: %s ===\n", all_pass ? "SEMUA TEST LOLOS" : "ADA TEST GAGAL");

    tensor_free(x); tensor_free(gain); tensor_free(out);
    tensor_free(xf); tensor_free(W1); tensor_free(W3); tensor_free(W2);
    tensor_free(outf); tensor_free(scratch1); tensor_free(scratch2); tensor_free(bad_out);

    return all_pass ? 0 : 1;
}
