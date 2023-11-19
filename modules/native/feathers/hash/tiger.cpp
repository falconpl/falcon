/* modified version (and adapted to falcon) of the Tiger reference implementation,
* containing some parts of the mhash C-lib Tiger implementation
*/
#include <string.h>
#include "tiger.h"


/* The following macro denotes that an optimization    */
/* for Alpha is required. It is used only for          */
/* optimization of time. Otherwise it does nothing.    */
// #ifdef __alpha
#define OPTIMIZE_FOR_ALPHA
// #endif

/* NOTE that this code is NOT FULLY OPTIMIZED for any  */
/* machine. Assembly code might be much faster on some */
/* machines, especially if the code is compiled with   */
/* gcc.                                                */

/* The number of passes of the hash function.          */
/* Three passes are recommended.                       */
/* Use four passes when you need extra security.       */
/* Must be at least three.                             */
#define PASSES 3

extern word64 tiger_table[4*256];

#define t1 (tiger_table)
#define t2 (tiger_table+256)
#define t3 (tiger_table+256*2)
#define t4 (tiger_table+256*3)

#define save_abc \
      aa = a; \
      bb = b; \
      cc = c;

#ifdef OPTIMIZE_FOR_ALPHA
/* This is the official definition of round */
#define round(a,b,c,x,mul) \
      c ^= x; \
      a -= t1[((c)>>(0*8))&0xFF] ^ t2[((c)>>(2*8))&0xFF] ^ \
	   t3[((c)>>(4*8))&0xFF] ^ t4[((c)>>(6*8))&0xFF] ; \
      b += t4[((c)>>(1*8))&0xFF] ^ t3[((c)>>(3*8))&0xFF] ^ \
	   t2[((c)>>(5*8))&0xFF] ^ t1[((c)>>(7*8))&0xFF] ; \
      b *= mul;
#else
/* This code works faster when compiled on 32-bit machines */
/* (but works slower on Alpha) */
#define round(a,b,c,x,mul) \
      c ^= x; \
      a -= t1[(byte)(c)] ^ \
           t2[(byte)(((word32)(c))>>(2*8))] ^ \
	   t3[(byte)((c)>>(4*8))] ^ \
           t4[(byte)(((word32)((c)>>(4*8)))>>(2*8))] ; \
      b += t4[(byte)(((word32)(c))>>(1*8))] ^ \
           t3[(byte)(((word32)(c))>>(3*8))] ^ \
	   t2[(byte)(((word32)((c)>>(4*8)))>>(1*8))] ^ \
           t1[(byte)(((word32)((c)>>(4*8)))>>(3*8))]; \
      b *= mul;
#endif

#define pass(a,b,c,mul) \
      round(a,b,c,x0,mul) \
      round(b,c,a,x1,mul) \
      round(c,a,b,x2,mul) \
      round(a,b,c,x3,mul) \
      round(b,c,a,x4,mul) \
      round(c,a,b,x5,mul) \
      round(a,b,c,x6,mul) \
      round(b,c,a,x7,mul)

#define key_schedule \
      x0 -= x7 ^ UI64LIT(0xA5A5A5A5A5A5A5A5); \
      x1 ^= x0; \
      x2 += x1; \
      x3 -= x2 ^ ((~x1)<<19); \
      x4 ^= x3; \
      x5 += x4; \
      x6 -= x5 ^ ((~x4)>>23); \
      x7 ^= x6; \
      x0 += x7; \
      x1 -= x0 ^ ((~x7)<<19); \
      x2 ^= x1; \
      x3 += x2; \
      x4 -= x3 ^ ((~x2)>>23); \
      x5 ^= x4; \
      x6 += x5; \
      x7 -= x6 ^ UI64LIT(0x0123456789ABCDEF);

#define feedforward \
      a ^= aa; \
      b -= bb; \
      c += cc;

#ifdef OPTIMIZE_FOR_ALPHA
/* The loop is unrolled: works better on Alpha */
#define compress \
      save_abc \
      pass(a,b,c,5) \
      key_schedule \
      pass(c,a,b,7) \
      key_schedule \
      pass(b,c,a,9) \
      for(pass_no=3; pass_no<PASSES; pass_no++) { \
        key_schedule \
	pass(a,b,c,9) \
	tmpa=a; a=c; c=b; b=tmpa;} \
      feedforward
#else
/* loop: works better on PC and Sun (smaller cache?) */
#define compress \
      save_abc \
      for(pass_no=0; pass_no<PASSES; pass_no++) { \
        if(pass_no != 0) {key_schedule} \
	pass(a,b,c,(pass_no==0?5:pass_no==1?7:9)); \
	tmpa=a; a=c; c=b; b=tmpa;} \
      feedforward
#endif

void tiger_init(tiger_ctx *ctx)
{
    ctx->blockcount = 0;
    ctx->index = 0;
    ctx->state[0]=UI64LIT(0x0123456789ABCDEF);
    ctx->state[1]=UI64LIT(0xFEDCBA9876543210);
    ctx->state[2]=UI64LIT(0xF096A5B4C3B2E187);
}

#ifndef EXTRACT_UCHAR
#define EXTRACT_UCHAR(p)  (*(byte *)(p))
#endif

#define STRING2INT64(s) ((((((((((((((word64)(EXTRACT_UCHAR(s+7) << 8)    \
    | EXTRACT_UCHAR(s+6)) << 8)  \
    | EXTRACT_UCHAR(s+5)) << 8)  \
    | EXTRACT_UCHAR(s+4)) << 8)  \
    | EXTRACT_UCHAR(s+3)) << 8)  \
    | EXTRACT_UCHAR(s+2)) << 8)  \
    | EXTRACT_UCHAR(s+1)) << 8)  \
    | EXTRACT_UCHAR(s))

static void tiger_block(struct tiger_ctx *ctx, const byte *str)
{
    word64 temp[8];
#ifdef BIG_ENDIAN
    word32 j;
    for(j=0; j<8; j++, str+=8)
        temp[j] = STRING2INT64(str);
    tiger_compress(temp, ctx->state);
#else
    memcpy(temp, str, 64); /* Required to avoid un-aligned access on some arches */
    tiger_compress(temp, ctx->state);
#endif
    /* Update block count */
    ctx->blockcount++;
}

/* The compress function is a function. Requires smaller cache?    */
void tiger_compress(word64 *str, word64 *state)
{
    word64 a, b, c, tmpa;
    word64 aa, bb, cc;
    word64 x0, x1, x2, x3, x4, x5, x6, x7;
    int pass_no;

    a = state[0];
    b = state[1];
    c = state[2];

    x0=str[0]; x1=str[1]; x2=str[2]; x3=str[3];
    x4=str[4]; x5=str[5]; x6=str[6]; x7=str[7];

    compress;

    state[0] = a;
    state[1] = b;
    state[2] = c;
}

void tiger_update(tiger_ctx *ctx, const byte *buffer, word32 len)
{
    word32 left;

    if (ctx->index) {	/* Try to fill partial block */
        left = 64 - ctx->index;
        if (len < left)
        {
            memcpy(ctx->block + ctx->index, buffer, len);
            ctx->index += len;
            return;	/* Finished */
        } else {
            memcpy(ctx->block + ctx->index, buffer, left);
            tiger_block(ctx, ctx->block);
            buffer += left;
            len -= left;
        }
    }
    while (len >= 64)
    {
        tiger_block(ctx, buffer);
        buffer += 64;
        len -= 64;
    }
    if ((ctx->index = len))
        /* This assignment is intended */
        /* Buffer leftovers */
        memcpy(ctx->block, buffer, len);
}

void tiger_finalize(tiger_ctx *ctx)
{
    word64 i, j;
    byte temp[64];
    i = ctx->index;
#ifdef BIG_ENDIAN
    for(j=0; j<i; j++)
        temp[j^7] = ctx->block[j];

    temp[j^7] = 0x01;
    j++;
    for(; j&7; j++)
        temp[j^7] = 0;
#else
    for(j=0; j<i; j++)
        temp[j] = ctx->block[j];

    temp[j++] = 0x01;
    for(; j&7; j++)
        temp[j] = 0;
#endif
    if(j>56)
    {
        for(; j<64; j++)
            temp[j] = 0;
        tiger_compress(((word64*)temp), ctx->state);
        j=0;
    }

    for(; j<56; j++)
        temp[j] = 0;
    ((word64*)(&(temp[56])))[0] = (ctx->blockcount << 9) + (ctx->index << 3);
    tiger_compress(((word64*)temp), ctx->state);
}

/* the output as presented at http://www.cs.technion.ac.il/~biham/Reports/Tiger/testresults.html */
void tiger_digest_little_endian_like(struct tiger_ctx *ctx, byte *s)
{
    word32 i;

    if (s)
        for (i = 0; i < 3; i++)
        {
            s[7] = byte(ctx->state[i]);
            s[6] = 0xff & (ctx->state[i] >> 8);
            s[5] = 0xff & (ctx->state[i] >> 16);
            s[4] = 0xff & (ctx->state[i] >> 24);
            s[3] = 0xff & (ctx->state[i] >> 32);
            s[2] = 0xff & (ctx->state[i] >> 40);
            s[1] = 0xff & (ctx->state[i] >> 48);
            s[0] = 0xff & (ctx->state[i] >> 56);
            s+=8;
        }
}

/* since Tiger does not have a specification of how the output is to be printed, we use the quasi-standard
 * (like MD5, SHA1, etc do it) of printing everything as big endian.
*/
void tiger_digest(struct tiger_ctx *ctx, byte *s)
{
    word32 i;

    if (s)
        for (i = 0; i < 3; i++)
        {
            s[0] = byte(ctx->state[i]);
            s[1] = 0xff & (ctx->state[i] >> 8);
            s[2] = 0xff & (ctx->state[i] >> 16);
            s[3] = 0xff & (ctx->state[i] >> 24);
            s[4] = 0xff & (ctx->state[i] >> 32);
            s[5] = 0xff & (ctx->state[i] >> 40);
            s[6] = 0xff & (ctx->state[i] >> 48);
            s[7] = 0xff & (ctx->state[i] >> 56);
            s+=8;
        }
}
