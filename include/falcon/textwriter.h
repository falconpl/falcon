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
 
    This class is meant to write text contained in Falcon strings. It uses a Transcorder
 to turn the internal Falcon character representation into output data respecting
 the requried encoding.

   Output is buffered autonomously by the Writer hierarcy; this means that using
 a buffered stream is sub-optimal, as the text encoding writer has its own
 buffering strategy.

 \see TextReader
 */

class FALCON_DYN_CLASS TextWriter: public Writer
{
public:
   /** Creates the text encoder using a standard "C" transcoder.
    \param stream The stream to be accessed.
    \param bOwn If true, the stream is closed and destroyed at reader destruction.
    */
   TextWriter( Stream* stream, bool bOwn = false );

   /** Creates the text encoder using determined transcoder.
    \param stream The stream to be accessed.
    \param decoder A text encoder obtained through Engine::getTranscoder.
    \param bOwn If true, the stream is closed and destroyed at reader destruction.
   */
   TextWriter( Stream* stream, Transcoder* decoder, bool bOwn = false );

   virtual ~TextWriter();

   /** Change the text decoder used by this reader.
    \param decoder the new decoder that should be used instead of the current one.

    This method allows to change the text decoder runtime.
    */
   void setEncoding( Transcoder* decoder );
   
   Transcoder* transcoder() const { return m_encoder; }

   /** Change automatic translation of LF into CRLF.

    Falcon defines the newline as '\\n' (ANSI/ASCII/UNICODE LF). On some OSs,
    or under some contexts (i.e. many internet protocols), 'lines' are separated
    by a CR + LF sequence.

    Instead of changing all the '\\n' into '\\r\\n in a text to be sent to a text
    file, it may be useful to use this facility that will scan independently for
    LF, and then write a CRLF in that position.

    The sequence is sent to the transcoder prior being written to the stream, so
    the CRLF sequence will be rendered correctly under the required encoding.

    This settings also affect the behavior of writeLine(); if set to true,
    writeLine will generate a CRLF sequence write after each call, instead of a
    simple LF character.

    Setting this facility slows down the writer operation. Where possible, i.e.
    if the need for the target stream to use CRLF sequences as line breaks is 
    known in advance, it may be advisable to manually use CRLF sequences as
    line separators on first instance.

    Use of this feature is suggested when the required behavior is not known
    when the content is created.

    \note writeLine() performance is not affected by this setting.
    */
   void setCRLF( bool mode ) { m_bCRLF = mode; }

   /** Set auto-flushing.
    \param mode True to flush the stream automatically after each end-of-line or
           after each writeLine() call.

    This method mimics the behavior of POSIX FILE* functions applied to process
    streams, which are flushed automatically at each end-of-line character.

    By default this setting is off.
    */
   void lineFlush( bool mode ) { m_bLineFlush = mode; }

   /** Checks if this writer is converting LF into CRLF sequences.
    \return true if LF->CRLF conversion is currently applied.
    */
   bool isCRLF() const { return m_bCRLF; }

   /** Checks if auto-flush at EOL is active.
    \return true if this stream automatically flushes at each EOL.
    */
   bool lineFlush() const { return m_bLineFlush; }
   
   /** Writes a text to a stream.
    \param str The string to be encoded on the stream.
    \param count The count of characters that must be read from the stream.
    \param start Start position where to put the decoded content in the target string.
    \return False if the string could be written -- and the underlying stream is not throwing errors.
    \throw EncodingError if the input stream has encoding errors.
    \throw IOError on I/O errors.

    On success, the contents of the string shall be completely transcoded.

    If setCRLF() method has been called, the input string is also searched for
    '\\n' (LF) characters, which are sent to the transcoder as CR+LF pairs.
    \note If a CRLF sequence is present in the original string, then the extra CR
    is \b not prepended to LF.

    If lineFlush() is set, then the underlying buffer is also flushed each time
    a LF character is met.
    */
   bool write( const String& str, length_t start = 0, length_t count = String::npos );

   /** Writes a text to a stream.
    \param str The string to be encoded on the stream.
    \param count The count of characters that must be read from the stream.
    \param start Start position where to put the decoded content in the target string.
    \return False if the string could be written -- and the underlying stream is not throwing errors.
    \throw EncodingError if the input stream has encoding errors.
    \throw IOError on I/O errors.

    On success, the contents of the string shall be completely transcoded.

    The output is considered to be a single line; this means that line feeds (LF)
    characters contained in the string are ignored. They are not translated into
    CRLF nor cause auto-flush to happen regardless of the setCRLF() and lineFlush()
    settings.

    After the content of the string is transcoded, a single end of line (LF or CRLF
    depending on the setCRLF setting) is written, and if lineFlush() is set then
    the writer is flushed prior return.
    */
   bool writeLine( const String& str, length_t start = 0, length_t count = String::npos );

   /** Writes a single character (UNICODE value) to a stream.
      \param chr The character to be written.
    \return false on stream I/O error (if not throwing exceptions).
    \throw IOError on I/O error in writing on underlying stream.
      
    If the character is an line-feed ('\\n', ANSI/ASCII/UNICODE 0x10) then the
    stream is flushed in case lineFlush() is set to true, and an extra CR character
    is sent to the transcoder if setCRLF is set to true.

    \note If a CRLF sequence is present in the original string, then the extra CR
    is \b not prepended to LF.
    */
   bool putChar( char_t chr );

protected:
   Transcoder* m_encoder;
   bool m_bWasCR;

   bool m_bCRLF;
   bool m_bLineFlush;
   String m_chrStr;

   size_t m_twBufSize;
   byte* m_twBuffer;

   bool rawWrite( const String& str, length_t start=0, length_t count=String::npos );
};

}

#endif	/* _FALCON_TEXTWRITER_H */

/* end of textwriter.h */
