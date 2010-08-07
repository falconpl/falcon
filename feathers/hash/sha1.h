#ifndef _SHA1_H
#define _SHA1_H

#include "hash_defs.h"

/* The SHA block size and message digest sizes, in bytes */

#define SHA_DATASIZE    64
#define SHA_DATALEN     16
#define SHA_DIGESTSIZE  20
#define SHA_DIGESTLEN    5
/* The structure for storing SHA info */

typedef struct sha_ctx {
  word32 digest[SHA_DIGESTLEN];  /* Message digest */
  word32 count_l, count_h;       /* 64-bit block count */
  byte block[SHA_DATASIZE];     /* SHA data buffer */
  word32 index;                  /* index into buffer */
} SHA_CTX;

void sha_init(struct sha_ctx *ctx);
void sha_update(struct sha_ctx *ctx, const byte *buffer, word32 len);
void sha_final(struct sha_ctx *ctx);
void sha_digest(struct sha_ctx *ctx, byte *s);
void sha_copy(struct sha_ctx *dest, struct sha_ctx *src);

#ifndef EXTRACT_UCHAR
#define EXTRACT_UCHAR(p)  (*(byte *)(p))
#endif

#define STRING2INT(s) ((((((EXTRACT_UCHAR(s) << 8)    \
			 | EXTRACT_UCHAR(s+1)) << 8)  \
			 | EXTRACT_UCHAR(s+2)) << 8)  \
			 | EXTRACT_UCHAR(s+3))
#endif
