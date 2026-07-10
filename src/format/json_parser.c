// json_parser.c
#include "json_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    const char *text;
    size_t pos;
    size_t len;
    int error;
} JsonParser;

static void skip_whitespace(JsonParser *p) {
    while (p->pos < p->len) {
        char c = p->text[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p->pos++;
        else break;
    }
}

static char peek(JsonParser *p) {
    if (p->pos >= p->len) return '\0';
    return p->text[p->pos];
}

static char advance(JsonParser *p) {
    if (p->pos >= p->len) return '\0';
    return p->text[p->pos++];
}

static JsonValue *json_alloc(JsonType type) {
    JsonValue *v = (JsonValue *)calloc(1, sizeof(JsonValue));
    v->type = type;
    return v;
}

static JsonValue *parse_value(JsonParser *p);

static char *parse_raw_string(JsonParser *p) {
    if (advance(p) != '"') { p->error = 1; return NULL; }

    size_t cap = 32, len = 0;
    char *buf = (char *)malloc(cap);

    while (p->pos < p->len && peek(p) != '"') {
        char c = advance(p);
        if (c == '\\') {
            char esc = advance(p);
            switch (esc) {
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case '/': c = '/'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case 'u': {
                    for (int i = 0; i < 4 && p->pos < p->len; i++) advance(p);
                    c = '?';
                    break;
                }
                default: c = esc; break;
            }
        }
        if (len + 1 >= cap) { cap *= 2; buf = (char *)realloc(buf, cap); }
        buf[len++] = c;
    }

    if (peek(p) != '"') { p->error = 1; free(buf); return NULL; }
    advance(p);

    buf[len] = '\0';
    return buf;
}

static JsonValue *parse_string(JsonParser *p) {
    char *s = parse_raw_string(p);
    if (!s) return NULL;
    JsonValue *v = json_alloc(JSON_STRING);
    v->as.string = s;
    return v;
}

static JsonValue *parse_number(JsonParser *p) {
    size_t start = p->pos;
    if (peek(p) == '-') advance(p);
    while (isdigit((unsigned char)peek(p))) advance(p);
    if (peek(p) == '.') {
        advance(p);
        while (isdigit((unsigned char)peek(p))) advance(p);
    }
    if (peek(p) == 'e' || peek(p) == 'E') {
        advance(p);
        if (peek(p) == '+' || peek(p) == '-') advance(p);
        while (isdigit((unsigned char)peek(p))) advance(p);
    }

    size_t len = p->pos - start;
    char *numbuf = (char *)malloc(len + 1);
    memcpy(numbuf, p->text + start, len);
    numbuf[len] = '\0';

    JsonValue *v = json_alloc(JSON_NUMBER);
    v->as.number = atof(numbuf);
    free(numbuf);
    return v;
}

static JsonValue *parse_literal(JsonParser *p, const char *literal, JsonValue *result) {
    size_t l = strlen(literal);
    if (p->pos + l > p->len || strncmp(p->text + p->pos, literal, l) != 0) {
        p->error = 1;
        return NULL;
    }
    p->pos += l;
    return result;
}

static JsonValue *parse_array(JsonParser *p) {
    advance(p);
    JsonValue *v = json_alloc(JSON_ARRAY);
    v->as.array.items = NULL;
    v->as.array.count = 0;

    skip_whitespace(p);
    if (peek(p) == ']') { advance(p); return v; }

    int cap = 4;
    v->as.array.items = (JsonValue **)malloc(sizeof(JsonValue *) * cap);

    while (1) {
        skip_whitespace(p);
        JsonValue *item = parse_value(p);
        if (!item) { p->error = 1; return v; }

        if (v->as.array.count >= cap) {
            cap *= 2;
            v->as.array.items = (JsonValue **)realloc(v->as.array.items, sizeof(JsonValue *) * cap);
        }
        v->as.array.items[v->as.array.count++] = item;

        skip_whitespace(p);
        char c = peek(p);
        if (c == ',') { advance(p); continue; }
        if (c == ']') { advance(p); break; }
        p->error = 1;
        break;
    }

    return v;
}

static JsonValue *parse_object(JsonParser *p) {
    advance(p);
    JsonValue *v = json_alloc(JSON_OBJECT);
    v->as.object.keys = NULL;
    v->as.object.values = NULL;
    v->as.object.count = 0;

    skip_whitespace(p);
    if (peek(p) == '}') { advance(p); return v; }

    int cap = 4;
    v->as.object.keys = (char **)malloc(sizeof(char *) * cap);
    v->as.object.values = (JsonValue **)malloc(sizeof(JsonValue *) * cap);

    while (1) {
        skip_whitespace(p);
        if (peek(p) != '"') { p->error = 1; break; }
        char *key = parse_raw_string(p);
        if (!key) { p->error = 1; break; }

        skip_whitespace(p);
        if (peek(p) != ':') { p->error = 1; free(key); break; }
        advance(p);

        skip_whitespace(p);
        JsonValue *val = parse_value(p);
        if (!val) { p->error = 1; free(key); break; }

        if (v->as.object.count >= cap) {
            cap *= 2;
            v->as.object.keys = (char **)realloc(v->as.object.keys, sizeof(char *) * cap);
            v->as.object.values = (JsonValue **)realloc(v->as.object.values, sizeof(JsonValue *) * cap);
        }
        v->as.object.keys[v->as.object.count] = key;
        v->as.object.values[v->as.object.count] = val;
        v->as.object.count++;

        skip_whitespace(p);
        char c = peek(p);
        if (c == ',') { advance(p); continue; }
        if (c == '}') { advance(p); break; }
        p->error = 1;
        break;
    }

    return v;
}

static JsonValue *parse_value(JsonParser *p) {
    skip_whitespace(p);
    char c = peek(p);

    if (c == '{') return parse_object(p);
    if (c == '[') return parse_array(p);
    if (c == '"') return parse_string(p);
    if (c == '-' || isdigit((unsigned char)c)) return parse_number(p);
    if (c == 't') { JsonValue *v = json_alloc(JSON_BOOL); v->as.boolean = 1; return parse_literal(p, "true", v); }
    if (c == 'f') { JsonValue *v = json_alloc(JSON_BOOL); v->as.boolean = 0; return parse_literal(p, "false", v); }
    if (c == 'n') { JsonValue *v = json_alloc(JSON_NULL); return parse_literal(p, "null", v); }

    p->error = 1;
    return NULL;
}

JsonValue *json_parse(const char *text) {
    if (!text) return NULL;
    JsonParser p = { text, 0, strlen(text), 0 };
    JsonValue *v = parse_value(&p);
    if (p.error) {
        fprintf(stderr, "[json_parse] error parsing di posisi %zu\n", p.pos);
        if (v) json_free(v);
        return NULL;
    }
    return v;
}

void json_free(JsonValue *v) {
    if (!v) return;
    switch (v->type) {
        case JSON_STRING:
            free(v->as.string);
            break;
        case JSON_ARRAY:
            for (int i = 0; i < v->as.array.count; i++) json_free(v->as.array.items[i]);
            free(v->as.array.items);
            break;
        case JSON_OBJECT:
            for (int i = 0; i < v->as.object.count; i++) {
                free(v->as.object.keys[i]);
                json_free(v->as.object.values[i]);
            }
            free(v->as.object.keys);
            free(v->as.object.values);
            break;
        default:
            break;
    }
    free(v);
}

JsonValue *json_object_get(const JsonValue *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJECT) return NULL;
    for (int i = 0; i < obj->as.object.count; i++) {
        if (strcmp(obj->as.object.keys[i], key) == 0) return obj->as.object.values[i];
    }
    return NULL;
}

double json_get_number(const JsonValue *v, double default_val) {
    if (!v || v->type != JSON_NUMBER) return default_val;
    return v->as.number;
}

int json_get_int(const JsonValue *v, int default_val) {
    if (!v || v->type != JSON_NUMBER) return default_val;
    return (int)(v->as.number + (v->as.number >= 0 ? 0.5 : -0.5));
}

const char *json_get_string(const JsonValue *v, const char *default_val) {
    if (!v || v->type != JSON_STRING) return default_val;
    return v->as.string;
}

int json_get_bool(const JsonValue *v, int default_val) {
    if (!v || v->type != JSON_BOOL) return default_val;
    return v->as.boolean;
}

int json_array_length(const JsonValue *v) {
    if (!v || v->type != JSON_ARRAY) return 0;
    return v->as.array.count;
}

JsonValue *json_array_get(const JsonValue *v, int index) {
    if (!v || v->type != JSON_ARRAY) return NULL;
    if (index < 0 || index >= v->as.array.count) return NULL;
    return v->as.array.items[index];
}
