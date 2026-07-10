// module.c
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "tensor.h"
#include "model.h"
#include "train.h"
#include "optimizer.h"
#include "tokenizer.h"
#include "pr_format.h"
#include <string.h>
#include <math.h>

static PyObject *py_train_and_save(PyObject *self, PyObject *args, PyObject *kwargs) {
    const char *text;
    const char *output_path;
    int dim = 32, heads = 4, head_dim = 8, ffn_hidden = 64, layers = 2;
    int vocab_size = 280, iters = 300, window = 32;
    double lr = 0.05;
    double decay_min = 0.7, decay_max = 0.95;

    static char *kwlist[] = {
        "text", "output_path", "dim", "heads", "head_dim", "ffn_hidden",
        "layers", "vocab_size", "iters", "lr", "window",
        "decay_min", "decay_max", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss|iiiiiiidiidd", kwlist,
            &text, &output_path, &dim, &heads, &head_dim, &ffn_hidden,
            &layers, &vocab_size, &iters, &lr, &window, &decay_min, &decay_max)) {
        return NULL;
    }

    if (dim != heads * head_dim) {
        PyErr_SetString(PyExc_ValueError, "dim harus sama dengan heads * head_dim");
        return NULL;
    }

    size_t text_len = strlen(text);

    BPETokenizer *tok = bpe_tokenizer_create(vocab_size);
    bpe_train(tok, (const unsigned char *)text, text_len);

    int *all_ids = NULL;
    int all_count = 0;
    bpe_encode(tok, (const unsigned char *)text, text_len, &all_ids, &all_count);

    if (all_count < 2) {
        bpe_tokenizer_free(tok);
        PyErr_SetString(PyExc_ValueError, "teks terlalu pendek buat training");
        return NULL;
    }

    ModelConfig cfg = {
        .vocab_size = tok->vocab_size, .dim = dim, .num_heads = heads,
        .head_dim = head_dim, .ffn_hidden = ffn_hidden, .num_layers = layers,
        .decay_min = decay_min, .decay_max = decay_max
    };
    Model *m = model_create(cfg, 42);
    if (!m) {
        bpe_tokenizer_free(tok);
        free(all_ids);
        PyErr_SetString(PyExc_RuntimeError, "gagal membuat model");
        return NULL;
    }

    int win = (all_count < window) ? all_count : window;
    int num_windows = all_count / win;
    if (num_windows < 1) num_windows = 1;

    Tensor **params;
    int num_params;
    model_collect_params(m, &params, &num_params);

    AdafactorState **states = (AdafactorState **)malloc(sizeof(AdafactorState *) * num_params);
    for (int i = 0; i < num_params; i++) states[i] = adafactor_state_create(params[i], 0.999, 1e-30, 1e-3);

    ModelCache *cache = model_cache_create(m, win);
    Tensor *logits = tensor_create((int[]){win, cfg.vocab_size}, 2, 0);
    Tensor *dLogits = tensor_create((int[]){win, cfg.vocab_size}, 2, 0);
    int *target_ids = (int *)malloc(sizeof(int) * (win - 1));

    double final_loss = 0.0;
    for (int iter = 0; iter < iters; iter++) {
        int w = iter % num_windows;
        int *window_ids = &all_ids[w * win];
        for (int i = 0; i < win - 1; i++) target_ids[i] = window_ids[i + 1];

        model_forward_train(m, window_ids, win, cache, logits);
        final_loss = cross_entropy_loss(logits, target_ids, win - 1, dLogits);

        model_zero_grad(m);
        model_backward(m, window_ids, win, cache, dLogits, 1e-6);

        for (int i = 0; i < num_params; i++) adafactor_step(params[i], states[i], lr);
    }

    int save_rc = pr_file_save(output_path, m, tok, 1);

    for (int i = 0; i < num_params; i++) adafactor_state_free(states[i]);
    free(states); free(params); free(all_ids); free(target_ids);
    tensor_free(logits); tensor_free(dLogits);
    model_cache_free(cache);
    model_free(m);
    bpe_tokenizer_free(tok);

    if (save_rc != 0) {
        PyErr_SetString(PyExc_IOError, "gagal menyimpan model");
        return NULL;
    }

    return PyFloat_FromDouble(final_loss);
}

static PyObject *py_load_and_generate(PyObject *self, PyObject *args, PyObject *kwargs) {
    const char *pr_path;
    const char *prompt;
    int max_tokens = 20;

    static char *kwlist[] = {"pr_path", "prompt", "max_tokens", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss|i", kwlist, &pr_path, &prompt, &max_tokens)) {
        return NULL;
    }

    Model *m = NULL;
    BPETokenizer *tok = NULL;
    int trained_flag = 0;
    if (pr_file_load(pr_path, &m, &tok, &trained_flag) != 0) {
        PyErr_SetString(PyExc_IOError, "gagal memuat file .pr");
        return NULL;
    }

    int *prompt_ids = NULL;
    int prompt_count = 0;
    bpe_encode(tok, (const unsigned char *)prompt, strlen(prompt), &prompt_ids, &prompt_count);

    int cap = prompt_count + max_tokens;
    int *ids = (int *)malloc(sizeof(int) * cap);
    memcpy(ids, prompt_ids, sizeof(int) * prompt_count);
    int count = prompt_count;
    free(prompt_ids);

    for (int step = 0; step < max_tokens; step++) {
        Tensor *logits = tensor_create((int[]){count, m->cfg.vocab_size}, 2, 0);
        if (model_forward(m, ids, count, logits) != 0) {
            tensor_free(logits);
            break;
        }
        double *last = &logits->data[(count - 1) * m->cfg.vocab_size];
        int best = 0; double best_val = last[0];
        for (int v = 1; v < m->cfg.vocab_size; v++) {
            if (last[v] > best_val) { best_val = last[v]; best = v; }
        }
        tensor_free(logits);

        if (count >= cap) break;
        ids[count++] = best;
    }

    unsigned char *decoded = NULL;
    size_t decoded_len = 0;
    bpe_decode(tok, ids, count, &decoded, &decoded_len);

    PyObject *result = PyUnicode_FromStringAndSize((const char *)decoded, decoded_len);

    free(ids);
    free(decoded);
    model_free(m);
    bpe_tokenizer_free(tok);

    return result;
}

static PyMethodDef PointRettaMethods[] = {
    {"train_and_save", (PyCFunction)py_train_and_save, METH_VARARGS | METH_KEYWORDS,
     "train_and_save(text, output_path, dim=32, heads=4, head_dim=8, ffn_hidden=64, "
     "layers=2, vocab_size=280, iters=300, lr=0.05, window=32, decay_min=0.7, decay_max=0.95) -> float\n"
     "Latih model PointRetta dari teks, simpan sebagai file .pr. Return loss akhir."},
    {"load_and_generate", (PyCFunction)py_load_and_generate, METH_VARARGS | METH_KEYWORDS,
     "load_and_generate(pr_path, prompt, max_tokens=20) -> str\n"
     "Muat model .pr, generate teks lanjutan dari prompt (greedy decoding)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pointretta_module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "PointRetta core C extension",
    -1,
    PointRettaMethods
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&pointretta_module);
}
