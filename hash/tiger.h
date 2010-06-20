#ifndef _TIGER_H_
#define _TIGER_H_

/* use falcon types and definitions */
#include "hash_defs.h"

struct tiger_ctx
{
    word64 state[3];
    word32 index;
    byte block[64];
    word64 blockcount;
};

void tiger_init(tiger_ctx *ctx);
void tiger_compress(word64 *str, word64 *state);
void tiger_update(tiger_ctx *ctx, const byte *buffer, word32 len);
void tiger_finalize(tiger_ctx *ctx);
void tiger_digest(struct tiger_ctx *ctx, byte *s);
void tiger_digest_little_endian_like(struct tiger_ctx *ctx, byte *s);


#endif
