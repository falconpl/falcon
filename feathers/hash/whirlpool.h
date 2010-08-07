#ifndef _AUTH_WHIRLPOOL_H
#define _AUTH_WHIRLPOOL_H


/*
* Whirlpool-specific definitions.
*/

#define DIGESTBYTES 64
#define DIGESTBITS  (8*DIGESTBYTES) /* 512 */

#define WBLOCKBYTES 64
#define WBLOCKBITS  (8*WBLOCKBYTES) /* 512 */

#define LENGTHBYTES 32
#define LENGTHBITS  (8*LENGTHBYTES) /* 256 */

#include <falcon/types.h>
typedef Falcon::uint8 uint8;
typedef Falcon::uint16 uint16;
typedef Falcon::uint32 uint32;
typedef Falcon::uint64 uint64;

struct whirlpool_ctx {
    uint8  bitLength[LENGTHBYTES]; /* global number of hashed bits (256-bit counter) */
    uint8  buffer[WBLOCKBYTES];	/* buffer of data to hash */
    uint32 bufferBits;		        /* current number of bits on the buffer */
    uint32 bufferPos;		        /* current (possibly incomplete) byte slot on the buffer */
    uint64 hash[DIGESTBYTES/8];    /* the hashing state */
};

void whirlpool_finalize(whirlpool_ctx *ctx, uint8 *result);
void whirlpool_update(const uint8 *source, uint32 sourceBits, whirlpool_ctx *ctx);
void whirlpool_init(whirlpool_ctx *ctx);

#endif
