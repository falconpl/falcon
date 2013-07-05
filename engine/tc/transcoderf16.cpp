/*
   FALCON - The Falcon Programming Language.
   FILE: transcoderf16.cpp

   Transcoder for text encodings - F16 Direct Falcon 16-bit encoding
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Mar 2011 13:04:14 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/tc/transcoderf16.h>
#include <falcon/stderrors.h>

#include <string.h>


namespace Falcon {

TranscoderF16::TranscoderF16():
   Transcoder( "F16" )
{
}

TranscoderF16::~TranscoderF16()
{
}

length_t TranscoderF16::encodingSize( length_t charCount ) const
{
   return charCount * 2;
}


length_t TranscoderF16::encode( const String& source, byte* data, length_t size,
            char_t undef, length_t start, length_t count ) const
{
   // check for a fast path.
   if( source.manipulator()->charSize() == 2 )
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
   uint16* tgt = reinterpret_cast<uint16*>(data);
   byte* enddata = data + size;

   while ( pos < end && reinterpret_cast<byte*>(tgt) < enddata )
   {
      char_t chr = source.getCharAt(pos++);
      if( chr > 0xFFFF )
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

   return (length_t)(reinterpret_cast<byte*>(tgt)-data);
}


length_t TranscoderF16::decode( const byte* data, length_t size, String& target, length_t count, bool ) const
{
   if( count*2 > size )
   {
      count = size/2;
   }
   
   target.manipulator( &csh::handler_buffer16 );
   target.reserve( count );  // already multiplied * 2
   target.size( count*2 );
   memcpy( target.getRawStorage(), data, count*2 );
   return count*2;
}
 
}

/* end of transcoderc.cpp */
