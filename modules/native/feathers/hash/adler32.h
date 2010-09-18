#ifndef _ADLER32_H_
#define _ADLER32_H_

/* use falcon types and definitions */
#include "hash_defs.h"

word32 adler32(word32 adler, char *buf, word32 len);

#endif
