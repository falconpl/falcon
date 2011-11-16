/*
   FALCON - The Falcon Programming Language
   FILE: base64.cpp

   Base64 encoding as per rfc3548
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Jun 2010 15:47:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/base64.h>
#include <falcon/autocstring.h>
#include <string.h>

namespace Falcon
{

bool Base64::encode( byte* data, uint32 dsize, byte* target, uint32& tgsize )
{
   static const char base64chars[] =
         "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

   // go the integer division point
   uint32 dsize_pair = (dsize/3) * 3;
   uint32 padCount = dsize - dsize_pair;

   if( tgsize < (dsize_pair / 3 * 4 + padCount)  )
   {
      // not enough space
      return false;
   }

   // perform everything up to the par
   uint32 rsize = 0;
   for (uint32 dp = 0; dp < dsize_pair; dp += 3)
   {
      // these three 8-bit (ASCII) characters become one 24-bit number
      uint32 b64val = data[dp] << 16;
      b64val += data[dp+1] << 8;
      b64val += data[dp+2];

      // this 24-bit number gets separated into four 6-bit numbers
      target[rsize++] = base64chars[(b64val >> 18) & 63];
      target[rsize++] = base64chars[(b64val >> 12) & 63];
      target[rsize++] = base64chars[(b64val >> 6) & 63];
      target[rsize++] = base64chars[b64val & 63];
   }

   // now add the last semi-four word
   if( padCount != 0 )
   {
      uint32 b64val = data[dsize_pair] << 16;
      // get enough bits
      if( (dsize_pair + 1) <  dsize )
         b64val += data[dsize_pair+1] << 8;
      if( (dsize_pair + 2) <  dsize )
         b64val += data[dsize_pair+2];

      // the first two semi-bytes are always in
      target[rsize++] = base64chars[(b64val >> 18) & 63];
      target[rsize++] = base64chars[(b64val >> 12) & 63];

      // then, the third only if we have an extra byte ...
      if( (dsize_pair + 1) <  dsize )
      {
         target[rsize++] = base64chars[(b64val >> 6) & 63];
         // and the fourth only if we have 2
         if( (dsize_pair + 2) <  dsize )
               target[rsize++] = base64chars[b64val & 63];
      }

      // fill the 1 or 2 missing chars with '='
      for (; padCount < 3; padCount++)
         target[rsize++] = '=';
   }

   // shrink the input buffer
   tgsize = rsize;
   return true;
}


bool Base64::decode( const String& data, byte* target, uint32& tgsize )
{
   uint32 tgpos = 0;
   uint32 dsize = data.size();

   uint32 i = 0;
   while( i < dsize )
   {
      uint32 value = 0;
      uint32 n = 0;
      while( n < 4 && i < dsize )
      {
         uint32 chr = data.getCharAt(i);

         // whitespace? -- ignore.
         if( chr == ' ' || chr == '\t' || chr == '\n' || chr == '\r' )
         {
            ++i;
            continue;
         }

         // padding char?
         if( chr == '=' )
         {
            ++i;
            while( i < dsize )
            {
               chr = data.getCharAt(i);
               if( ! (chr == '=' || chr == ' ' || chr == '\t' || chr == '\n' || chr == '\r') )
               {
                  // we had a "=" not at the end, this is not a base64 string.
                  return false;
               }
               ++i;
            }

            // ok, finish this off.
            break;
         }

         uint32 bits = getBits( chr );

         if( bits == 0xFF )
            return false;

         // create the value
         value |= bits << ((3-n)*6);

         // advance in the string and in the status.
         ++i;
         ++n;
      }

      // get the chars
      if( (tgpos+n-1) >= tgsize )
         return false;

      // have we to fix something?
      if( n > 0 )
      {
         target[tgpos++] = (byte)((value>>16) &0xFF);

         if( n > 1 )
         {
            target[tgpos++] = (byte)((value>>8) &0xFF);

            if( n > 2 )
               target[tgpos++] = (byte)(value &0xFF);
         }
      }
   }

   tgsize = tgpos;
   return true;
}


byte Base64::getBits( uint32 chr )
{
   if( chr >= 'A' && chr <= 'Z' )
   {
     return (byte)( chr - 'A');
   }
   else if ( chr >= 'a' && chr <= 'z' )
   {
     return (byte)(chr - 'a' + 26);
   }
   else if ( chr >= '0' && chr <= '9' )
   {
     return (byte)(chr - '0' + 52);
   }
   else if ( chr == '+' )
   {
     return 62;
   }
   else if( chr == '/' )
   {
     return 63;
   }
   else
   {
     // It's not a valid base64 encoding
     return 0xFF;
   }
}


void Base64::encode( byte* data, uint32 dsize, String& target )
{
   uint32 tgsize = dsize/3*4+4;
   target.reserve( tgsize );
   encode( data, dsize, target.getRawStorage(), tgsize );
   target.size( tgsize );
}


void Base64::encode( const String& data, String& target )
{
   AutoCString cdata( data );
   encode( (byte*) cdata.c_str(), cdata.length(), target );
}


bool Base64::decode( const String& data, String& target )
{
   // enough space to store a correctly encoded base64 string.
   String tg2;
   uint32 tgsize = data.size()/4*3+4;
   tg2.reserve( tgsize );
   if( decode( data, tg2.getRawStorage(), tgsize ) )
   {
      tg2.size( tgsize );
      tg2.c_ize();
      return target.fromUTF8( (char*)tg2.getRawStorage() );
   }

   return false;
}

}
