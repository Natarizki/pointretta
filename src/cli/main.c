// main.c
// PointRetta CLI entry point. Dispatches to subcommand: build, train, delete.

#include "cmd_build.h"
#include "cmd_train.h"
#include "cmd_delete.h"
#include <stdio.h>
#include <string.h>

static void print_usage(void) {
    printf("PointRetta CLI\n\n");
    printf("Usage:\n");
    printf("  pointretta build config.json name.json   Build a model skeleton (.prtm)\n");
    printf("  pointretta train file.prtm dataset...    Train a model (.prtm -> .pr)\n");
    printf("  pointretta delete file.pr                Delete a model file\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const char *command = argv[1];

    if (strcmp(command, "build") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: pointretta build config.json name.json\n");
            return 1;
        }
        return cmd_build(argv[2], argv[3]) == 0 ? 0 : 1;
    }

    if (strcmp(command, "train") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: pointretta train file.prtm dataset1 [dataset2 ...]\n");
            return 1;
        }
        return cmd_train(argv[2], &argv[3], argc - 3) == 0 ? 0 : 1;
    }

    if (strcmp(command, "delete") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: pointretta delete file.pr\n");
            return 1;
        }
        return cmd_delete(argv[2]) == 0 ? 0 : 1;
    }

    fprintf(stderr, "Unknown command: %s\n\n", command);
    print_usage();
    return 1;
}
