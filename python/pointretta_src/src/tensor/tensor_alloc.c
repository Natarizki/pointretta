// tensor_alloc.c
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void compute_strides(const int *shape, int ndim, int *strides_out) {
    int acc = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides_out[i] = acc;
        acc *= shape[i];
    }
}

Tensor *tensor_create(const int *shape, int ndim, int requires_grad) {
    if (ndim <= 0 || ndim > TENSOR_MAX_DIMS) {
        fprintf(stderr, "[tensor_create] ndim tidak valid: %d\n", ndim);
        return NULL;
    }

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) {
        fprintf(stderr, "[tensor_create] gagal alokasi struct Tensor\n");
        return NULL;
    }

    t->ndim = ndim;
    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        total *= (size_t)shape[i];
    }
    for (int i = ndim; i < TENSOR_MAX_DIMS; i++) {
        t->shape[i] = 1;
    }

    compute_strides(t->shape, ndim, t->strides);
    for (int i = ndim; i < TENSOR_MAX_DIMS; i++) {
        t->strides[i] = 0;
    }

    t->size = total;

    t->data = (double *)calloc(total, sizeof(double));
    if (!t->data) {
        fprintf(stderr, "[tensor_create] gagal alokasi data (%zu elemen)\n", total);
        free(t);
        return NULL;
    }

    t->requires_grad = requires_grad;
    if (requires_grad) {
        t->grad = (double *)calloc(total, sizeof(double));
        if (!t->grad) {
            fprintf(stderr, "[tensor_create] gagal alokasi grad (%zu elemen)\n", total);
            free(t->data);
            free(t);
            return NULL;
        }
    } else {
        t->grad = NULL;
    }

    return t;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    if (t->data) free(t->data);
    if (t->grad) free(t->grad);
    free(t);
}

void tensor_fill(Tensor *t, double value) {
    if (!t || !t->data) return;
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = value;
    }
}

void tensor_fill_random(Tensor *t, double min, double max, unsigned int seed) {
    if (!t || !t->data) return;
    srand(seed);
    double range = max - min;
    for (size_t i = 0; i < t->size; i++) {
        double r = (double)rand() / (double)RAND_MAX;
        t->data[i] = min + r * range;
    }
}

int tensor_copy(Tensor *dst, const Tensor *src) {
    if (!dst || !src || !dst->data || !src->data) return -1;
    if (dst->size != src->size) {
        fprintf(stderr, "[tensor_copy] size tidak cocok: dst=%zu src=%zu\n",
                dst->size, src->size);
        return -1;
    }
    memcpy(dst->data, src->data, dst->size * sizeof(double));
    return 0;
}

void tensor_zero_grad(Tensor *t) {
    if (!t || !t->grad) return;
    memset(t->grad, 0, t->size * sizeof(double));
}

size_t tensor_index(const Tensor *t, const int *coords) {
    size_t idx = 0;
    for (int i = 0; i < t->ndim; i++) {
        idx += (size_t)coords[i] * (size_t)t->strides[i];
    }
    return idx;
}

double tensor_get(const Tensor *t, const int *coords) {
    size_t idx = tensor_index(t, coords);
    return t->data[idx];
}

void tensor_set(Tensor *t, const int *coords, double value) {
    size_t idx = tensor_index(t, coords);
    t->data[idx] = value;
}

void tensor_print(const Tensor *t, const char *name) {
    if (!t) {
        printf("%s: (NULL tensor)\n", name);
        return;
    }
    printf("%s: shape=[", name);
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->shape[i], (i < t->ndim - 1) ? "," : "");
    }
    printf("] size=%zu requires_grad=%d\n", t->size, t->requires_grad);

    if (t->size <= 64) {
        printf("  data: [");
        for (size_t i = 0; i < t->size; i++) {
            printf("%.5f%s", t->data[i], (i < t->size - 1) ? ", " : "");
        }
        printf("]\n");
    } else {
        printf("  data: (terlalu besar buat ditampilkan, %zu elemen)\n", t->size);
    }
}
