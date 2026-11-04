#include "pipette_llama.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// The app's half of pipette_llama.h: loads one llama.cpp build per process and defines
// every call Swift makes, each forwarding to that build's table. Swift chooses the build
// (`LlamaPins.load`) before a cell measures anything; a call made before any choice loads
// main, which is what tests and the UI run.
// ---------------------------------------------------------------------------

static pthread_mutex_t pipette_llama_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic(const struct pipette_llama_table *) pipette_llama_current;
static char pipette_llama_current_pin[PIPETTE_LLAMA_PIN_MAX + 1];

static const struct pipette_llama_table * pipette_llama_open(const char * pin, char * err,
                                                             size_t err_len) {
    if (strlen(pin) > PIPETTE_LLAMA_PIN_MAX) {
        snprintf(err, err_len, "the pin name %s is longer than %d bytes", pin,
                 PIPETTE_LLAMA_PIN_MAX);
        return NULL;
    }
    char path[2 * PIPETTE_LLAMA_PIN_MAX + 32];
    snprintf(path, sizeof path, "@rpath/%s.framework/%s", pin, pin);
    // RTLD_LOCAL keeps the build's symbols out of every other image's lookups.
    void * image = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!image) {
        snprintf(err, err_len, "%s", dlerror());
        return NULL;
    }
    pipette_llama_table_fn table_of = (pipette_llama_table_fn) dlsym(image, PIPETTE_LLAMA_TABLE_SYMBOL);
    if (!table_of) {
        snprintf(err, err_len, "%s exports no %s", path, PIPETTE_LLAMA_TABLE_SYMBOL);
        dlclose(image);
        return NULL;
    }
    const struct pipette_llama_table * table = table_of();
    if (table->version != PIPETTE_LLAMA_TABLE_VERSION || table->size != sizeof *table) {
        snprintf(err, err_len, "%s has table version %u (%u bytes), the app needs %u (%zu bytes)",
                 path, table->version, table->size, PIPETTE_LLAMA_TABLE_VERSION, sizeof *table);
        dlclose(image);
        return NULL;
    }
    return table;
}

const struct pipette_llama_table * pipette_llama_load(const char * pin, char * err, size_t err_len) {
    pthread_mutex_lock(&pipette_llama_lock);
    const struct pipette_llama_table * table = atomic_load(&pipette_llama_current);
    if (table) {
        if (strcmp(pin, pipette_llama_current_pin) != 0) {
            snprintf(err, err_len, "this process already runs %s, and a loaded build cannot be unloaded",
                     pipette_llama_current_pin);
            table = NULL;
        }
    } else if ((table = pipette_llama_open(pin, err, err_len))) {
        snprintf(pipette_llama_current_pin, sizeof pipette_llama_current_pin, "%s", pin);
        atomic_store(&pipette_llama_current, table);
    }
    pthread_mutex_unlock(&pipette_llama_lock);
    return table;
}

const struct pipette_llama_table * pipette_llama_probe(const char * pin, char * err, size_t err_len) {
    return pipette_llama_open(pin, err, err_len);
}

const char * pipette_llama_loaded_pin(void) {
    return atomic_load(&pipette_llama_current) ? pipette_llama_current_pin : NULL;
}

static const struct pipette_llama_table * pipette_llama(void) {
    const struct pipette_llama_table * table = atomic_load(&pipette_llama_current);
    if (table) {
        return table;
    }
    char err[512];
    table = pipette_llama_load(PIPETTE_LLAMA_MAIN_PIN, err, sizeof err);
    if (!table) {
        // Main ships in every build of the app, so this is a broken bundle.
        fprintf(stderr, "pipette: cannot load %s: %s\n", PIPETTE_LLAMA_MAIN_PIN, err);
        abort();
    }
    return table;
}

#define PIPETTE_LLAMA_FORWARD_R(ret, name, params, args) \
    ret name params { return pipette_llama()->name args; }
#define PIPETTE_LLAMA_FORWARD_V(name, params, args) \
    void name params { pipette_llama()->name args; }
PIPETTE_LLAMA_CALLS(PIPETTE_LLAMA_FORWARD_R, PIPETTE_LLAMA_FORWARD_V)
