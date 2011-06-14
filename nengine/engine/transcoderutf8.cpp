/*
   FALCON - The Falcon Programming Language.
   FILE: transcoderutf8.cpp

   Transcoder for text encodings - UTF-8
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 Mar 2011 23:30:43 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/transcoderutf8.h>
#include <falcon/encodingerror.h>

namespace Falcon {

TranscoderUTF8::TranscoderUTF8():
   Transcoder( "utf8" )
{
}

TranscoderUTF8::~TranscoderUTF8()
{
}

length_t TranscoderUTF8::encodingSize( length_t charCount ) const
{
   return charCount*4;
}

length_t TranscoderUTF8::encode( const String& source, byte* data, length_t size,
            char_t undef, length_t start, length_t count ) const
{
   length_t end;

   if ( count == String::npos || count+start > source.length() )
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
   byte* enddata1 = data + size-1;
   byte* enddata2 = data + size-2;
   byte* enddata3 = data + size-3;

   while( pos < end && tgt < enddata )
   {
      char_t chr = source.getCharAt(pos);

      if ( chr < 0x80 )
      {
         *tgt = (byte) chr;
      }
      // two byes range?
      else if ( chr < 0x800 )
      {
         // have we enough space to encode this?
         if( tgt >= enddata1 )
         {
            return (length_t)(tgt - data);
         }

         *tgt = 0xC0 | ((chr >> 6 ) & 0x1f); ++tgt;
         *tgt = 0x80 | (0x3f & chr);
      }
      else if ( chr < 0x10000 )
      {
         // have we enough space to encode this?
         if( tgt >= enddata2 )
         {
            return (length_t)(tgt - data);
         }

         *tgt = 0xE0 | ((chr >> 12) & 0x0f ); ++tgt;
         *tgt = 0x80 | ((chr >> 6) & 0x3f ); ++tgt;
         *tgt = 0x80 | (0x3f & chr);
      }
      else
      {
         // have we enough space to encode this?
         if( tgt >= enddata3 )
         {
            return (length_t)(tgt - data);
         }

         *tgt = 0xF0 | ((chr >> 18) & 0x7 ); ++tgt;
         *tgt = 0x80 | ((chr >> 12) & 0x3f ); ++tgt;
         *tgt = 0x80 | ((chr >> 6) & 0x3f ); ++tgt;
         *tgt = 0x80 | (0x3f & chr);
      }

      ++tgt;
      ++pos;
   }

   return (length_t)(tgt - data);
}

length_t TranscoderUTF8::decode( const byte* data, length_t size, String& target, length_t count, bool bThrow ) const
{
   const byte* utf8 = data;
   const byte* limit = data+count;

   length_t done = 0;
   target.reserve( count );  // can be max this size.
   target.size(0);

   while( utf8 < limit && done < count )
   {
      char_t chr = 0;

      byte in = (byte) *utf8;

      // 4 bytes? -- pattern 1111 0xxx
      int bits;
      if ( (in & 0xF8) == 0xF0 )
      {
         chr = (in & 0x7 ) << 18;

         if ( utf8 + 3 >= limit )
         {
            // we can't decode this -- return what we have decoded up to date
            return (length_t)(utf8-data);
         }

         bits = 18;
      }
      // 3 bytes -- pattern 1110 xxxx
      else if ( (in & 0xF0) == 0xE0 )
      {
         chr = (in & 0xF) << 12;

         if ( utf8 + 2 >= limit )
         {
            // we can't decode this -- return what we have decoded up to date
            return (length_t)(utf8-data);
         }

         bits = 12;
      }
      // 2 bytes -- pattern 110x xxxx
      else if ( (in & 0xE0) == 0xC0 )
      {
         chr = (in & 0x1F) << 6;

         if ( utf8 + 1 >= limit )
         {
            // we can't decode this -- return what we have decoded up to date
            return (length_t)(utf8-data);
         }

         bits = 6;
      }
      else if( in < 0x80 )
      {
         chr = (char_t) in;
         bits = 0;
      }
      // invalid pattern
      else {
         if( bThrow )
         {
            throw new EncodingError(ErrorParam(e_enc_fail, __LINE__, __FILE__)
                  .extra(name()) );
         }
         
         return (char_t) -1;
      }

      // read the other characters with pattern 0x10xx xxxx
      while( bits > 0 )
      {
         bits -= 6;

         // we know we have enough space, or we would have returned.
         ++utf8;
         byte in = (byte) *utf8;

         if( (in & 0xC0) != 0x80 )
         {
            // protocol error
            if( bThrow )
            {
               throw new EncodingError(ErrorParam(e_enc_fail, __LINE__, __FILE__)
                     .extra(name()) );
            }

            return (char_t)-1;
         }

         chr |= (in & 0x3f) << bits;
      }

      // save the character
      target.append( chr );
      ++done;
      ++utf8;
   }

   // return the amount of data that has been transcoded.
   return (length_t)(utf8-data);
}

}

/* transcoderutf8.cpp */
