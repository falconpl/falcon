/*
   FALCON - The Falcon Programming Language.
   FILE: ht_types.h

   Declaration of types used by Falcon language.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-05-15

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_TYPES_H
#define _FALCON_TYPES_H

#include <falcon/setup.h>

// Inclusion of stddef for size_t
#include <stddef.h>
#include <sys/types.h>

namespace Falcon
{

class VMachine;
class VMContext;
class Module;

typedef char *   cstring;

typedef unsigned char byte;
typedef unsigned char * bytearray;

typedef byte uint8;
typedef unsigned short int uint16;
typedef unsigned int uint32;

#ifdef _MSC_VER
typedef unsigned __int64 uint64;
#else
typedef unsigned long long int uint64;
#endif

typedef char int8;
typedef short int int16;
typedef int int32;

#ifdef _MSC_VER
typedef __int64 int64;
#else
typedef long long int int64;
#endif

typedef double numeric;
typedef void * voidp;

// length used in all the sizes
typedef uint32 length_t;
typedef uint32 char_t;

//#if !defined(off_t) && !defined(_OFF_T)
//#define off_t int64
//#endif
typedef int64 off_t;


typedef void (CDECL *ext_func_t) ( VMContext *ctx, int32 pCount );

extern "C" {
   typedef Module* (CDECL  *ext_mod_init)();
}

}

#endif
/* end of ht_types.h */
