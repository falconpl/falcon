#if !defined(__MHASH_MD4_H)
#define __MHASH_MD4_H

/* use falcon types and definitions */
#include "hash_defs.h"

struct MD4_CTX {
	word32 buf[4];
	word32 bits[2];
	union {byte bytes[64]; word32 words[16];}  in;
};

void MD4Init(struct MD4_CTX *context);
void MD4Update(struct MD4_CTX *context, const byte *buf, word32 len);
void MD4Final( struct MD4_CTX *context, byte *digest);
void MD4Transform(word32 *buf, word32 *in);


#endif /* !__MHASH_MD4_H */
