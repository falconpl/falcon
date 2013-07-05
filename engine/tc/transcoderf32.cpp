/*
   FALCON - The Falcon Programming Language.
   FILE: transcoderf32.cpp

   Transcoder for text encodings -
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Mar 2011 13:04:14 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/tc/transcoderf32.h>
#include <falcon/stderrors.h>

#include <string.h>


namespace Falcon {

TranscoderF32::TranscoderF32():
   Transcoder( "C" )
{
}

TranscoderF32::~TranscoderF32()
{
}

length_t TranscoderF32::encodingSize( length_t charCount ) const
{
   return charCount*4;
}


length_t TranscoderF32::encode( const String& source, byte* data, length_t size,
            char_t, length_t start, length_t count ) const
{
   // check for a fast path.
   if( source.manipulator()->charSize() == 4 )
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
   uint32* tgt = reinterpret_cast<uint32*>(data);
   byte* enddata = data + size;

   while ( pos < end && reinterpret_cast<byte*>(tgt) < enddata )
   {
      char_t chr = source.getCharAt(pos++);
      *tgt = chr;
      ++tgt;
   }

   return (length_t)(reinterpret_cast<byte*>(tgt)-data);
}


length_t TranscoderF32::decode( const byte* data, length_t size, String& target, length_t count, bool ) const
{
   if( count*4 > size )
   {
      count = size/4;
   }
   
   target.manipulator( &csh::handler_buffer );
   target.reserve( count );  // can be max this size.
   target.size( count*4 );
   memcpy( target.getRawStorage(), data, count*4 );
   return count*4;
}
 
}

/* end of transcoderc.cpp */
