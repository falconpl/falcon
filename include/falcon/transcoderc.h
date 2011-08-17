/*
   FALCON - The Falcon Programming Language.
   FILE: transcoderc.h

   Transcoder for text encodings - C (neuter encoding)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 Mar 2011 23:30:43 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_TRANSCODERC_H
#define	_FALCON_TRANSCODERC_H

#include <falcon/transcoder.h>

namespace Falcon {

/** Text Decoder for the C encoding (neuter - null encoding).

 C Encoding is defined as an as-is 8-bit encoding. The 7-bit range is granted
 to be ASCII/UNICODE lower page, while the meaning of the characters in the
 range 128-255 (0x80-0xFF) is undefined and system-specific.

*/
class FALCON_DYN_CLASS TranscoderC: public Transcoder
{

public:
   TranscoderC();
   virtual ~TranscoderC();

   virtual length_t encode( const String& source, byte* data, length_t size,
            char_t undef = '?', length_t start=0, length_t count = String::npos ) const;

   virtual length_t encodingSize( length_t charCount ) const;

   /** Decodes a null-encoded text.

    C Encoding can never fail, and will always encode the required "count" bytes.
    This is done by setting the target to a buffer 8bit string and performing a raw
    copy of the input data.
    */
   virtual length_t decode( const byte* data, length_t size, String& target, length_t count, bool bThrow = true ) const ;
};

}

#endif	/* _FALCON_TEXTDECODERC_H */

/* end of transcoderc.h */
