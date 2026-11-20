// LD_PRELOAD logger for /dev/mpp_service MPP_IOC_CFG_V1 requests (debug oracle only)
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/ioctl.h>

typedef struct { uint32_t cmd, flag, size, offset; uint64_t data_ptr; } Req;
#define MPP_IOC_CFG_V1 _IOW('v', 1, unsigned int)
static FILE *out;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
typedef struct { int fd; uint32_t off, size; uint32_t *ptr; } Rd;
static Rd rds[256]; static int nrd;
static uint64_t seen[64]; static int nseen;

static void dump(const uint32_t *w, uint32_t size, uint32_t off) {
    for (uint32_t i = 0; i < size / 4; i++) {
        if (i % 8 == 0) fprintf(out, "\n  %05x:", off + i * 4);
        fprintf(out, " %08x", w[i]);
    }
    fprintf(out, "\n");
}

int ioctl(int fd, unsigned long request, ...) {
    va_list ap; va_start(ap, request); void *arg = va_arg(ap, void *); va_end(ap);
    static int (*real)(int, unsigned long, void *);
    if (!real) real = dlsym(RTLD_NEXT, "ioctl");
    if (!out) { const char *p = getenv("SHIM_LOG"); out = fopen(p ? p : "/tmp/mppshim.log", "w"); }
    if (request != MPP_IOC_CFG_V1) {
        pthread_mutex_lock(&mu);
        int known = 0; for (int i = 0; i < nseen; i++) known |= seen[i] == request;
        if (!known && nseen < 64) { seen[nseen++] = request; fprintf(out, "OTHER fd=%d req=%lx\n", fd, request); fflush(out); }
        pthread_mutex_unlock(&mu);
        return real(fd, request, arg);
    }
    Req *r = arg; int is_poll = 0;
    pthread_mutex_lock(&mu);
    for (int i = 0;; i++) {
        fprintf(out, "REQ fd=%d i=%d cmd=%x flag=%x size=%u off=%x", fd, i, r[i].cmd, r[i].flag, r[i].size, r[i].offset);
        if (r[i].cmd == 0x201 && nrd < 256) { rds[nrd++] = (Rd){fd, r[i].offset, r[i].size, (uint32_t *)(uintptr_t)r[i].data_ptr}; }
        if (r[i].cmd == 0x300 || r[i].cmd == 0x301) is_poll = 1;
        if (r[i].data_ptr && r[i].size && r[i].cmd != 0x201) dump((uint32_t *)(uintptr_t)r[i].data_ptr, r[i].size, r[i].offset);
        else fprintf(out, "\n");
        if (!(r[i].flag & 1) || (r[i].flag & 2)) break;
    }
    fflush(out);
    pthread_mutex_unlock(&mu);
    int ret = real(fd, request, arg);
    pthread_mutex_lock(&mu);
    if (r[0].cmd < 0x100 && r[0].data_ptr) fprintf(out, "QRET fd=%d cmd=%x ret=%d val=%08x\n", fd, r[0].cmd, ret, *(uint32_t *)(uintptr_t)r[0].data_ptr);
    if (is_poll) {
        fprintf(out, "POLLRET fd=%d ret=%d\n", fd, ret);
        int keep = 0;
        for (int i = 0; i < nrd; i++) {
            if (rds[i].fd == fd) { fprintf(out, "RD fd=%d", fd); dump(rds[i].ptr, rds[i].size, rds[i].off); }
            else rds[keep++] = rds[i];
        }
        nrd = keep;
    }
    fflush(out);
    pthread_mutex_unlock(&mu);
    return ret;
}
