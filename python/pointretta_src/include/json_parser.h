// json_parser.h
#ifndef POINTRETTA_JSON_PARSER_H
#define POINTRETTA_JSON_PARSER_H

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

typedef struct JsonValue {
    JsonType type;
    union {
        int boolean;
        double number;
        char *string;
        struct { struct JsonValue **items; int count; } array;
        struct { char **keys; struct JsonValue **values; int count; } object;
    } as;
} JsonValue;

JsonValue *json_parse(const char *text);
void json_free(JsonValue *v);

JsonValue *json_object_get(const JsonValue *obj, const char *key);
double json_get_number(const JsonValue *v, double default_val);
int json_get_int(const JsonValue *v, int default_val);
const char *json_get_string(const JsonValue *v, const char *default_val);
int json_get_bool(const JsonValue *v, int default_val);
int json_array_length(const JsonValue *v);
JsonValue *json_array_get(const JsonValue *v, int index);

#endif // POINTRETTA_JSON_PARSER_H
