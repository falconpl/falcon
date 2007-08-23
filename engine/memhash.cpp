/*
   FALCON - The Falcon Programming Language.
   FILE: memhash.cpp
   $Id: memhash.cpp,v 1.2 2006/10/11 16:14:59 gian Exp $

   Calculates checksum + hash value of a memory range
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago 23 23:01:46 CEST 2004

   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
