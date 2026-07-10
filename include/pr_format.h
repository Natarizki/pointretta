// pr_format.h
#ifndef POINTRETTA_PR_FORMAT_H
#define POINTRETTA_PR_FORMAT_H

#include "model.h"
#include "tokenizer.h"

int pr_file_save(const char *path, const Model *m, const BPETokenizer *tok, int trained_flag);

int pr_file_load(const char *path, Model **out_model, BPETokenizer **out_tok, int *out_trained_flag);

#endif // POINTRETTA_PR_FORMAT_H
