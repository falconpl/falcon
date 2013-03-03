/*
   FALCON - The Falcon Programming Language.
   FILE: transcoderf16.h

   Transcoder for text encodings - Falcon String direct 16-bit
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 Mar 2011 23:30:43 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_TRANSCODERF16_H
#define  _FALCON_TRANSCODERF16_H

#include <falcon/transcoder.h>

namespace Falcon {

/** Text Enc-Decoder for 16-bit Falcon Strings.
   Source of this transcoder is considered to be a Falcon string
   with charSize() == 4.

   It is internally used by the functions that create readers out
   of strings.
*/
class FALCON_DYN_CLASS TranscoderF16: public Transcoder
{

public:
   TranscoderF16();
   virtual ~TranscoderF16();

   virtual length_t encode( const String& source, byte* data, length_t size,
            char_t undef = '?', length_t start=0, length_t count = String::npos ) const;

   virtual length_t encodingSize( length_t charCount ) const;
   virtual length_t decode( const byte* data, length_t size, String& target, length_t count, bool bThrow = true ) const ;
};

}

#endif

/* end of transcoderf16.h */
