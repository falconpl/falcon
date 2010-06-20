#ifndef _SHA1_H_
#define _SHA1_H_

/* use falcon types and definitions */
#include "hash_defs.h"

struct SHA1_CTX
{
    word32 state[5];
    word32 count[2];
    byte buffer[64];
};


void SHA1Transform(word32 *state, const byte *buffer);
void SHA1Init(SHA1_CTX *context);
void SHA1Update(SHA1_CTX *context, const byte *data, word32 len);
void SHA1Final(byte *digest, SHA1_CTX *context);

#endif
