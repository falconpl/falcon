/*
   FALCON - The Falcon Programming Language.
   FILE: transcoderutf8.h

   Transcoder for text encodings - UTF-8
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 Mar 2011 23:30:43 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_TRANSCODERUTF8_H
#define	_FALCON_TRANSCODERUTF8_H

#include <falcon/transcoder.h>

namespace Falcon {
/** Text Decoder for the UTF-8 encoding. */
class FALCON_DYN_CLASS TranscoderUTF8: public Transcoder
{

public:
   TranscoderUTF8();
   virtual ~TranscoderUTF8();
   virtual length_t encodingSize( length_t charCount ) const;
   virtual length_t encode( const String& source, byte* data, length_t size,
            char_t undef = '?', length_t start=0, length_t count = String::npos ) const;
   virtual length_t decode( const byte* data, length_t size, String& target, length_t count, bool bThrow = true ) const;
};

}

#endif	/* _FALCON_TRANSCODERUTF8_H */

/* end of transcoderutf8.h */
