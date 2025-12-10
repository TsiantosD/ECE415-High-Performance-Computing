#ifndef HELPERS_H
#define HELPERS_H

PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

#endif