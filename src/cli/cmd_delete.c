// cmd_delete.c
// Implementation of "pointretta delete file.pr" with re-type-to-confirm safety.

#include "cmd_delete.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *get_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

int cmd_delete(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: file '%s' not found\n", path);
        return -1;
    }
    fclose(f);

    const char *fname = get_basename(path);

    printf("WARNING: You are about to PERMANENTLY delete '%s'.\n", path);
    printf("This action cannot be undone.\n\n");
    printf("Type the filename again to confirm (%s): ", fname);
    fflush(stdout);

    char input[512];
    if (!fgets(input, sizeof(input), stdin)) {
        printf("\nCancelled (no input).\n");
        return -1;
    }

    size_t len = strlen(input);
    while (len > 0 && (input[len - 1] == '\n' || input[len - 1] == '\r')) {
        input[--len] = '\0';
    }

    if (strcmp(input, fname) != 0) {
        printf("Name does not match. Cancelled, file was NOT deleted.\n");
        return -1;
    }

    if (remove(path) != 0) {
        fprintf(stderr, "Error: failed to delete '%s'\n", path);
        return -1;
    }

    printf("File '%s' deleted successfully.\n", path);
    return 0;
}
