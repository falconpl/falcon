#ifndef _SHA1_H_
#define _SHA1_H_

struct SHA1_CTX
{
    unsigned long state[5];
    unsigned long count[2];
    unsigned char buffer[64];
};


void SHA1Transform(unsigned long *state, unsigned char *buffer);
void SHA1Init(SHA1_CTX *context);
void SHA1Update(SHA1_CTX *context, unsigned char *data, unsigned int len);
void SHA1Final(unsigned char *digest, SHA1_CTX *context);

#endif
