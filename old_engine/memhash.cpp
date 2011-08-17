/*
   FALCON - The Falcon Programming Language.
   FILE: memhash.cpp

   Calculates checksum + hash value of a memory range
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago 23 23:01:46 CEST 2004


   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/common.h>
#include <falcon/string.h>

namespace Falcon {

uint32 calcMemHash( const char *memory, uint32 size )
{
   uint32 sum = flc_HASH_SEED;
   int i = 2;
   for( const char *p = memory; p < memory + size; p++, i++ ) {
      sum += static_cast<byte>( *p ) * i;
   }
   return sum;
}

uint32 calcCstrHash( const char *cstring )
{
   uint32 sum = flc_HASH_SEED;
   int i = 2;
   for( const char *p = cstring; *p != 0; p++, i++ ) {
      sum += static_cast<byte>( *p ) * i;
   }
   return sum;
}

uint32 calcStringHash( const String &str )
{
   uint32 sum = flc_HASH_SEED;
   int i = 2;
	uint32 len = str.length();
   for( uint32 pos = 0; pos < len ; pos++, i++ ) {
      sum += str.getCharAt( pos ) * i;
   }
   return sum;
}

}

/* end of memhash.cpp */
