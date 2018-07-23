#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
double second_linux (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
