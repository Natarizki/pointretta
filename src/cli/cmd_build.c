// cmd_build.c
// Implementation of "pointretta build config.json name.json"
// Reads config.json + name.json, builds a ModelConfig, creates a skeleton
// model (random init) + skeleton tokenizer (base vocab 256, no merges yet),
// saves it as <name>.prtm

#include "cmd_build.h"
#include "json_parser.h"
#include "model.h"
#include "tokenizer.h"
#include "pr_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char *read_file_to_string(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open file '%s'\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buf = (char *)malloc(size + 1);
    if (!buf) { fclose(f); return NULL; }

    size_t read_size = fread(buf, 1, size, f);
    buf[read_size] = '\0';
    fclose(f);
    return buf;
}

int cmd_build(const char *config_path, const char *name_path) {
    printf("PointRetta build\n");
    printf("  config: %s\n", config_path);
    printf("  name:   %s\n\n", name_path);

    char *config_text = read_file_to_string(config_path);
    if (!config_text) return -1;

    JsonValue *config = json_parse(config_text);
    free(config_text);
    if (!config) {
        fprintf(stderr, "Error: failed to parse %s (invalid JSON)\n", config_path);
        return -1;
    }

    char *name_text = read_file_to_string(name_path);
    if (!name_text) { json_free(config); return -1; }

    JsonValue *name_cfg = json_parse(name_text);
    free(name_text);
    if (!name_cfg) {
        fprintf(stderr, "Error: failed to parse %s (invalid JSON)\n", name_path);
        json_free(config);
        return -1;
    }

    JsonValue *arch = json_object_get(config, "architecture");
    JsonValue *retention = json_object_get(config, "retention");

    ModelConfig cfg;
    cfg.dim = json_get_int(json_object_get(arch, "dim"), 256);
    cfg.num_layers = json_get_int(json_object_get(arch, "layers"), 4);
    cfg.num_heads = json_get_int(json_object_get(arch, "heads"), 4);
    cfg.head_dim = json_get_int(json_object_get(arch, "head_dim"), cfg.dim / cfg.num_heads);
    cfg.ffn_hidden = json_get_int(json_object_get(arch, "ffn_hidden"), cfg.dim * 4);
    cfg.vocab_size = json_get_int(json_object_get(arch, "vocab_size"), 300);
    cfg.decay_min = json_get_number(json_object_get(retention, "decay_min"), 0.7);
    cfg.decay_max = json_get_number(json_object_get(retention, "decay_max"), 0.99);

    if (cfg.dim != cfg.num_heads * cfg.head_dim) {
        fprintf(stderr, "Error: dim (%d) must equal heads*head_dim (%d*%d=%d)\n",
                cfg.dim, cfg.num_heads, cfg.head_dim, cfg.num_heads * cfg.head_dim);
        json_free(config); json_free(name_cfg);
        return -1;
    }

    printf("Architecture:\n");
    printf("  dim=%d layers=%d heads=%d head_dim=%d ffn_hidden=%d vocab_size=%d\n",
           cfg.dim, cfg.num_layers, cfg.num_heads, cfg.head_dim, cfg.ffn_hidden, cfg.vocab_size);
    printf("  decay_min=%.3f decay_max=%.3f\n\n", cfg.decay_min, cfg.decay_max);

    const char *model_name = json_get_string(json_object_get(name_cfg, "name"), "unnamed-model");
    printf("Model name: %s\n\n", model_name);

    unsigned int seed = (unsigned int)time(NULL);
    Model *m = model_create(cfg, seed);
    if (!m) {
        fprintf(stderr, "Error: failed to create model (check configuration)\n");
        json_free(config); json_free(name_cfg);
        return -1;
    }

    BPETokenizer *tok = bpe_tokenizer_create(cfg.vocab_size);
    if (!tok) {
        fprintf(stderr, "Error: failed to create tokenizer\n");
        model_free(m);
        json_free(config); json_free(name_cfg);
        return -1;
    }

    char output_path[512];
    snprintf(output_path, sizeof(output_path), "%s.prtm", model_name);

    int save_rc = pr_file_save(output_path, m, tok, 0);
    if (save_rc != 0) {
        fprintf(stderr, "Error: failed to save %s\n", output_path);
        model_free(m); bpe_tokenizer_free(tok);
        json_free(config); json_free(name_cfg);
        return -1;
    }

    printf("Successfully created: %s\n", output_path);
    printf("Next step: pointretta train %s <dataset>\n", output_path);

    model_free(m);
    bpe_tokenizer_free(tok);
    json_free(config);
    json_free(name_cfg);

    return 0;
}
