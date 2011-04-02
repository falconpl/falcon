/*
   FALCON - The Falcon Programming Language.
   FILE: transcoderc.cpp

   Transcoder for text encodings - C (neuter) encoding
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Mar 2011 13:04:14 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/transcoderc.h>
#include <string.h>

#include "falcon/encodingerror.h"

namespace Falcon {

TranscoderC::TranscoderC():
   Transcoder( "C" )
{
}

TranscoderC::~TranscoderC()
{
}

length_t TranscoderC::encodingSize( length_t charCount ) const
{
   return charCount;
}


length_t TranscoderC::encode( const String& source, byte* data, length_t size,
            char_t undef, length_t start, length_t count ) const
{
   // check for a fast path.
   if( source.manipulator()->charSize() == 1 )
   {
      // easy...
      if ( count + start > source.size() || count == String::npos )
      {
         count = source.size() - start;
      }

      if ( count > size )
      {
         count = size;
      }

      memcpy( data, source.getRawStorage() + start, count );

      return count;
   }

   // ... multiple byte characters; management is a bit more complex.
   length_t end;

   if ( count+start > source.length() || count == String::npos )
   {
      end = source.length()-start;
   }
   else
   {
      end = start + count;
   }

   length_t pos = start;
   byte* tgt = data;
   byte* enddata = data + size;

   while ( pos < end && tgt < enddata )
   {
      char_t chr = source.getCharAt(pos++);
      if( chr > 0xFF )
      {
         if (undef > 0)
         {
            *tgt = undef;
         }
         else
         {
            throw new EncodingError(ErrorParam(e_enc_fail, __LINE__, __FILE__).extra("C") );
         }
      }
      else
      {
         *tgt = chr;
      }

      ++tgt;
   }

   return (length_t)(tgt-data);
}


length_t TranscoderC::decode( const byte* data, length_t size, String& target, length_t count, bool ) const
{
   if( count > size )
   {
      count = size;
   }
   
   target.manipulator( &csh::handler_buffer );
   target.reserve( count );  // can be max this size.
   target.size( count );
   memcpy( target.getRawStorage(), data, count );
   return count;
}
 
}

/* end of transcoderc.cpp */
