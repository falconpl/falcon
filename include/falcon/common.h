/*
   FALCON - The Falcon Programming Language.
   FILE: flc_common.h
   $Id: common.h,v 1.4 2007/06/08 18:20:08 jonnymind Exp $

   Definition for falcon common library.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom giu 20 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/


#ifndef flc_COMMON_H
#define flc_COMMON_H

#include <falcon/types.h>

namespace Falcon
{

class String;

#define flc_CURRENT_VER  1
#define flc_CURRENT_SUB  0


#define flcC_SYM_VAR   0
#define flcC_SYM_FUNC  1
#define flcC_SYM_EXT   2
#define flcC_SYM_CLASS 3
#define flcC_SYM_VARPROP   4
#define flcC_EXPORT_BIT 0x80

#define flcC_VAL_NIL 0
#define flcC_VAL_INT 1
#define flcC_VAL_NUM 2
#define flcC_VAL_STRID 3
#define flcC_VAL_SIMID 4

/** Seed for the hash key checksum generator */
#define flc_HASH_SEED      0xC2AF3DE4


/** Utility class to char pointers.

   This class is just an operator that compares the strings pointed by its parameters.
   It is used in some maps that have a string pointer as they key, as it points a string
   being the name of a symbol or other kind of string whose instance must be kept somewhere
   else.
*/
class CharPtrCmp
{
   public:
      bool operator() ( const char *s1, const char *s2 ) const
         {
            while( *s1 && *s2 && *s1 == *s2 ) {
               s1 ++;
               s2 ++;
            }
            return (*s1 < *s2);
         }
};

#define flc_ASM_GLOBAL   0
#define flc_ASM_LOCAL    1
#define flc_ASM_PARAM    2


#if FALCON_LITTLE_ENDIAN == 1

inline uint64 endianInt64( const uint64 param ) { return param; }
inline uint32 endianInt32( const uint32 param ) { return param; }
inline uint16 endianInt16( const uint16 param ) { return param; }
inline numeric endianNum( const numeric param ) { return param; }
#else
inline uint64 endianInt64( const uint64 param ) {
   byte *chars = (byte *) &param;
   return ((uint64)chars[0]) << 56 | ((uint64)chars[1]) << 48 | ((uint64)chars[2]) << 40 |
          ((uint64)chars[3]) << 32 | ((uint64)chars[4]) << 24 | ((uint64)chars[5]) << 16 |
          ((uint64)chars[6]) << 8 | ((uint64)chars[7]);
}

inline numeric endianNum( const numeric param ) {
   byte *chars = (byte *) &param;
   return ((uint64)chars[0]) << 56 | ((uint64)chars[1]) << 48 | ((uint64)chars[2]) << 40 |
          ((uint64)chars[3]) << 32 | ((uint64)chars[4]) << 24 | ((uint64)chars[5]) << 16 |
          ((uint64)chars[6]) << 8 | ((uint64)chars[7]);
}

inline uint32 endianInt32( const uint32 param ) {
   byte *chars = (byte *) &param;
   return ((uint32)chars[0]) << 24 | ((uint32)chars[1]) << 16 | ((uint32)chars[2]) << 8 | ((uint32)chars[3]);
}

inline uint16 endianInt16( const uint16 param ) {
   byte *chars = (byte *) &param;
   return ((uint32)chars[0]) << 8 | ((uint32)chars[1]);
}

#endif /* FALCON_LITTLE_ENDIAN */

inline int charToHex( const char elem )
{
   if( elem >= '0' && elem <= '9' )
      return elem - '0';
   else if( elem >= 'A' && elem <= 'F' )
     return elem - 'A';
   else if( elem >= 'a' && elem <= 'f' )
      return elem - 'a';

   return -1;
}

FALCON_DYN_SYM uint32 calcMemHash( const char *memory, uint32 size );
FALCON_DYN_SYM uint32 calcCstrHash( const char *cstring );
FALCON_DYN_SYM uint32 calcStringHash( const String &string );
FALCON_DYN_SYM inline uint32 calcIntHash( const int32 number ) { return flc_HASH_SEED * number; }

}

#endif

/* end of flc_common.h */
