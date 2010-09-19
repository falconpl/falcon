/* sha256_sha224.h
 *
 * The sha256 and sha224 hash functions.
 */

/* nettle, low-level cryptographics library
 *
 * Copyright (C) 2001 Niels Müller
 *  
 * The nettle library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at your
 * option) any later version.
 * 
 * The nettle library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with the nettle library; see the file COPYING.LIB.  If not, write to
 * the Free Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
 * MA 02111-1307, USA.
 */

/* The SHA224 support was added by B. Poettering in April 2004. */
 
#if !defined(__MHASH_SHA256_SHA224_H_INCLUDED)
#define __MHASH_SHA256_SHA224_H_INCLUDED

/* use falcon types and definitions */
#include "hash_defs.h"

/* SHA256 and SHA224 */

#define SHA256_DIGEST_SIZE 32
#define SHA224_DIGEST_SIZE 28
#define SHA256_SHA224_DATA_SIZE 64

/* State is kept internally as 8 32-bit words. */
#define _SHA256_SHA224_DIGEST_LENGTH 8

struct sha256_sha224_ctx
{
  word32 state[_SHA256_SHA224_DIGEST_LENGTH]; /* State variables */
  word64 bitcount;                            /* Bit counter */
  byte block[SHA256_SHA224_DATA_SIZE];        /* SHA256/224 data buffer */
  word32 index;                               /* index into buffer */
};

void
sha256_init(struct sha256_sha224_ctx *ctx);

void
sha224_init(struct sha256_sha224_ctx *ctx);

void
sha256_sha224_update(struct sha256_sha224_ctx *ctx, const byte *data, word32 length);

void
sha256_sha224_final(struct sha256_sha224_ctx *ctx);

void
sha256_digest(struct sha256_sha224_ctx *ctx, byte *digest);

void
sha224_digest(struct sha256_sha224_ctx *ctx, byte *digest);


#endif /* __MHASH_SHA256_SHA224_H_INCLUDED */
