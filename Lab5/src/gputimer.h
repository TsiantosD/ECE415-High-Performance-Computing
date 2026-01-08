#ifndef GPUTIMER_H
#define GPUTIMER_H
#include "helpers.h"
#include <sys/time.h>

static struct timeval timerStart, timerStop;

void create_timer() {

}

void destroy_timer() {

}

void start_timer() {
    gettimeofday(&timerStart, NULL);
}

void stop_timer() {
    gettimeofday(&timerStop, NULL);
}

float get_timer_ms() {
    struct timeval timerElapsed;
    timersub(&timerStop, &timerStart, &timerElapsed);
    return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
}

#endif

