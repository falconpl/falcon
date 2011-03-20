/*
   FALCON - The Falcon Programming Language.
   FILE: textwriter.h

   Text-oriented stream reader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Mar 2011 11:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_TEXTWRITER_H
#define	_FALCON_TEXTWRITER_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/writer.h>

namespace Falcon {

class Stream;
class Transcoder;

/** Class providing the ability to read data from a text oriented stream.
 
    This class is meant to read text into strings. It uses a text decoder
 to transform raw data input into characters under a given encoding.

   Input is buffered autonomously by the Reader hierarcy; this means that using
 a buffered stream is sub-optimal, as the text encoding reader has its own
 buffering strategy.

 \see TextReader
 */

class FALCON_DYN_CLASS TextWriter: public Writer
{
public:
   /** Creates the text encoder using a standard "C" text decoder.
    \param stream The stream to be accessed.
    \param bOwn If true, the stream is closed and destroyed at reader destruction.
    */
   TextWriter( Stream* stream, bool bOwn = false );

   /** Creates the text encoder using determined transcoder.
    \param stream The stream to be accessed.
    \param decoder A text decoder obtained through Engine::getTextDecoder.
    \param bOwn If true, the stream is closed and destroyed at reader destruction.
   */
   TextWriter( Stream* stream, Transcoder* decoder, bool bOwn = false );

   virtual ~TextWriter();

   /** Change the text decoder used by this reader.
    \param decoder the new decoder that should be used instead of the current one.

    This method allows to change the text decoder runtime.
    */
   void setEncoding( Transcoder* decoder );

   /** Writes a text to a stream.
    \param str The string to be encoded on the stream.
    \param count The count of characters that must be read from the stream.
    \param start Start position where to put the decoded content in the target string.
    \return True if the string could be written.
    \throw EncodingError if the input stream has encoding errors.

    */
   bool write( String& str, length_t start = 0, length_t count = 0 );

   bool writeLine( String& str, bool bFlush = true, length_t start = 0, length_t count = 0 );

   void putChar( char_t chr );

protected:
   Transcoder* m_encoder;
};

}

#endif	/* _FALCON_TEXTWRITER_H */

/* end of textwriter.h */
