/*
   FALCON - The Falcon Programming Language.
   FILE: textdecoder.h

   Transcoder for text encodings - ABC.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 Mar 2011 23:30:43 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_TRANSCODER_H
#define	_FALCON_TRANSCODER_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>

namespace Falcon {
/** Transforms incoming bytes into abstract Falcon characters and strings.
 This abstract base class is the base of the text decoder hierarcy.

 Text decoders are registered with the engine and can be accessed through
 the Engine::getTextDecoder() method.

 Text decoders operate on pre.allocated buffers as data input, and on strings
 as data output.

 They provide a relatively basic interface, meant to convert the data in the
 input buffer into a meaningful sequence of characters in the target string.

 */
class FALCON_DYN_CLASS Transcoder
{
public:
   virtual ~Transcoder();

   /** Returns the amount of bytes needed to encode a string of the given size.
    \param charCount The size of a string to be encoded in characters.
    \return Number of bytes that can be assumed to safely store an encoded string.

    This method is meant to determine the maximum size of a target buffer that should
    be spared in advance to store a string encoded in the required format. Variable
    size encodings should provide the maximum expected size.
    */
   virtual length_t encodingSize( length_t charCount ) const = 0;

   /** Encodes a string on a data buffer.
    \param source The string to be encoded.
    \param data The target buffer where to encode the string.
    \param size Size of the target buffer.
    \param undef Character used to substitute characters that cannot be encoded
           through this encoding.
    \param start First character of the string to be encoded.
    \param count Number of characters in the string to be encoded.
    \return Amount of bytes written in the target buffer.
    \throw EncodingError if undef is set to 0, in case an unencodeable character
    is found.

    The size of the target buffer should be large enough to store the encoded text.
    The required amount of data can be knwon in advance by calling encodingSize().
    If the buffer is not large enough, the method should simply fill it up to where
    possible and then return.

    */
   virtual length_t encode( const String& source, byte* data, length_t size,
            char_t undef = '?', length_t start=0, length_t count = String::npos ) const = 0;

   /** Decodes a buffer.
    \param data Input buffer containing the encoded data.
    \param size The size of the buffer to be decoded.
    \param target Where to store the decoded text.
    \param count Maximum number of characters to be read.
    \param bThrow If true, throw an exception in case of incorrect encoding.
    \return the number of bytes that could be decoded, or String::npos if the data is not
    correctly encoded (if bThrow if false).

    This method tries to tecode at maximum count bytes in the data stream. The
    method must be prepared to decode partial data, and eventually return 0 if
    nothing from the input data can be decoded.

    Conversely, if it's possible to decide that the data is not correctly encoded,
    i.e. becuase complete in size but incorrect in contents, the method shall
    throw an EncodingError.

    At times it is useful to test for an encoding by repeatedly applying different
    decoders 'till one is found correct; in that case, it's possible to pass bThrow
    false to return String::npos instead of throwing an exception.

    \note The target string is cleared before starting the decoding.
    */
   virtual length_t decode( const byte* data, length_t size, String& target, length_t count, bool bThrow = true ) const = 0;
   
   /** Return the ANSI description of the text encoding provided by this encoder. */
   const String& name() const { return m_name; }

protected:
   String m_name;

   /** Constructs the decoder.
    \param name the ANSI name for the text encoding.
   */
   Transcoder( const String& encName );
};

}

#endif	/* _FALCON_TRANSCODER_H */

/* end of textdecoder.h */
