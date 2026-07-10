// test_json_parser.c
#include "json_parser.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    printf("=== Test 1: Parse config.json PointRetta ===\n");
    const char *config_json =
        "{"
        "  \"architecture\": {"
        "    \"dim\": 512,"
        "    \"layers\": 12,"
        "    \"heads\": 8"
        "  },"
        "  \"retention\": {"
        "    \"multi_scale\": true,"
        "    \"decay_min\": 0.9,"
        "    \"decay_max\": 0.999"
        "  },"
        "  \"tokenizer\": {"
        "    \"special_tokens\": [\"<pad>\", \"<bos>\", \"<eos>\", \"<unk>\"]"
        "  },"
        "  \"quantization\": {"
        "    \"enabled\": false"
        "  }"
        "}";

    JsonValue *config = json_parse(config_json);
    if (!config) {
        printf("GAGAL: json_parse return NULL\n");
        return 1;
    }

    JsonValue *arch = json_object_get(config, "architecture");
    int dim = json_get_int(json_object_get(arch, "dim"), -1);
    int layers = json_get_int(json_object_get(arch, "layers"), -1);
    int heads = json_get_int(json_object_get(arch, "heads"), -1);
    printf("architecture.dim = %d (harusnya 512)\n", dim);
    printf("architecture.layers = %d (harusnya 12)\n", layers);
    printf("architecture.heads = %d (harusnya 8)\n", heads);

    JsonValue *retention = json_object_get(config, "retention");
    int multi_scale = json_get_bool(json_object_get(retention, "multi_scale"), 0);
    double decay_min = json_get_number(json_object_get(retention, "decay_min"), -1.0);
    double decay_max = json_get_number(json_object_get(retention, "decay_max"), -1.0);
    printf("retention.multi_scale = %d (harusnya 1)\n", multi_scale);
    printf("retention.decay_min = %.3f (harusnya 0.900)\n", decay_min);
    printf("retention.decay_max = %.3f (harusnya 0.999)\n", decay_max);

    JsonValue *tokenizer = json_object_get(config, "tokenizer");
    JsonValue *special_tokens = json_object_get(tokenizer, "special_tokens");
    int num_tokens = json_array_length(special_tokens);
    printf("tokenizer.special_tokens count = %d (harusnya 4)\n", num_tokens);
    printf("  isi: [");
    for (int i = 0; i < num_tokens; i++) {
        const char *tok = json_get_string(json_array_get(special_tokens, i), "?");
        printf("%s%s", tok, (i < num_tokens - 1) ? ", " : "");
    }
    printf("]\n");

    JsonValue *quant = json_object_get(config, "quantization");
    int quant_enabled = json_get_bool(json_object_get(quant, "enabled"), 1);
    printf("quantization.enabled = %d (harusnya 0)\n", quant_enabled);

    int test1_pass = (dim == 512) && (layers == 12) && (heads == 8) &&
                      (multi_scale == 1) && (decay_min == 0.9) && (decay_max == 0.999) &&
                      (num_tokens == 4) && (quant_enabled == 0);
    printf("\nTest 1: %s\n\n", test1_pass ? "LOLOS" : "GAGAL");

    json_free(config);

    printf("=== Test 2: Parse name.json PointRetta ===\n");
    const char *name_json =
        "{"
        "  \"name\": \"my-first-model\","
        "  \"version\": \"0.1.0\","
        "  \"author\": \"Nata\","
        "  \"tags\": [\"experimental\", \"indonesian\"],"
        "  \"huggingface\": {"
        "    \"enabled\": true,"
        "    \"repo\": \"\","
        "    \"private\": false"
        "  }"
        "}";

    JsonValue *name_cfg = json_parse(name_json);
    const char *name = json_get_string(json_object_get(name_cfg, "name"), "?");
    const char *version = json_get_string(json_object_get(name_cfg, "version"), "?");
    const char *author = json_get_string(json_object_get(name_cfg, "author"), "?");

    printf("name = \"%s\" (harusnya \"my-first-model\")\n", name);
    printf("version = \"%s\" (harusnya \"0.1.0\")\n", version);
    printf("author = \"%s\" (harusnya \"Nata\")\n", author);

    JsonValue *hf = json_object_get(name_cfg, "huggingface");
    int hf_enabled = json_get_bool(json_object_get(hf, "enabled"), 0);
    const char *hf_repo = json_get_string(json_object_get(hf, "repo"), "MISSING");
    int hf_private = json_get_bool(json_object_get(hf, "private"), 1);
    printf("huggingface.enabled = %d (harusnya 1)\n", hf_enabled);
    printf("huggingface.repo = \"%s\" (harusnya kosong)\n", hf_repo);
    printf("huggingface.private = %d (harusnya 0)\n", hf_private);

    int test2_pass = (strcmp(name, "my-first-model") == 0) &&
                      (strcmp(version, "0.1.0") == 0) &&
                      (strcmp(author, "Nata") == 0) &&
                      (hf_enabled == 1) && (strcmp(hf_repo, "") == 0) && (hf_private == 0);
    printf("\nTest 2: %s\n\n", test2_pass ? "LOLOS" : "GAGAL");

    json_free(name_cfg);

    printf("=== Test 3: Edge case (angka negatif, nested array, key hilang) ===\n");
    const char *edge_json = "{\"a\": -3.14, \"b\": [[1,2],[3,4]], \"c\": null}";
    JsonValue *edge = json_parse(edge_json);
    double a = json_get_number(json_object_get(edge, "a"), 0.0);
    JsonValue *b = json_object_get(edge, "b");
    JsonValue *b0 = json_array_get(b, 0);
    int b0_1 = json_get_int(json_array_get(b0, 1), -1);
    JsonValue *missing = json_object_get(edge, "missing_key");
    double missing_default = json_get_number(missing, 99.0);

    printf("a = %.2f (harusnya -3.14)\n", a);
    printf("b[0][1] = %d (harusnya 2)\n", b0_1);
    printf("missing_key (pakai default) = %.1f (harusnya 99.0)\n", missing_default);

    int test3_pass = (a == -3.14) && (b0_1 == 2) && (missing_default == 99.0);
    printf("\nTest 3: %s\n\n", test3_pass ? "LOLOS" : "GAGAL");

    json_free(edge);

    int all_pass = test1_pass && test2_pass && test3_pass;
    printf("=== HASIL AKHIR: %s ===\n", all_pass ? "SEMUA TEST LOLOS" : "ADA TEST GAGAL");
    return all_pass ? 0 : 1;
}
