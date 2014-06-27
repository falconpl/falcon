/*
   FALCON - The Falcon Programming Language.
   FILE: textreader.h

   Text-oriented stream reader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Mar 2011 11:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_TEXTREADER_H
#define	_FALCON_TEXTREADER_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/reader.h>

namespace Falcon {

class Stream;
class Transcoder;

/** Class providing the ability to read data from a text oriented stream.
 
    This class is meant to read text into strings. It uses a text decoder
 to transform raw data input into characters under a given encoding.

   Input is buffered autonomously by the Reader hierarcy; this means that using
 a buffered stream is sub-optimal, as the text encoding reader has its own
 buffering strategy.

 \note As the buffering policy is reader-specific, a stream delegated to a reader
 cannot be used atonomously after it is being used here. However, it is possible
 to re-delegate the same stream to another reader, maintaning its status and
 pre-read buffers, to change the way it is accessed.

 A typical pattern usage of a text reader is that of scanning a network stream
 for HTTP-like headers, and then turn into a raw data reader to get all the
 incoming data. This can be performed through the following pattern:

 @code
 SocketStream s( ... ); // intialize the stream -- can be stack owned.
 TextReader headReader( &s );

 int clength = 0;
 // ... read the heder lines lines ...
 String header;
 while (  headReader.readline( header ) && header != "" )
 {
    //... parse the headers ...
 }

 if ( clength )
 {
 DataReader contentReader;
 headReader.delegate( contentReader );
 buffer = new byte[clength];
 contentReader.read( buffer, clength );
 }
 // ...

 @endcode

 \note TextReader methods never enlarge the character size of the string, which
   is automatically enlarged when the first character that would not fit in the
   current character size is met. Is performance is to be preferred to space constraint,
   it is wiser to provide a String whose character size has been enlarged to the maximum
   size that the desired text encoding may support.
 */

class FALCON_DYN_CLASS TextReader: public Reader
{
public:
   static const char_t NoChar = (char_t)-1;
   
   /** Creates the text decoder using a standard "C" transcoder.
    \param stream The stream to be accessed.
    \param bOwn If true, the stream is closed and destroyed at reader destruction.
    */
   TextReader( Stream* stream );

   /** Creates the text decoder using determined text decoder.
    \param stream The stream to be accessed.
    \param decoder A transcoder obtained through Engine::getTranscoder.
    \param bOwn If true, the stream is closed and destroyed at reader destruction.
   */
   TextReader( Stream* stream, Transcoder* decoder );

   virtual ~TextReader();

   /** Change the text decoder used by this reader.
    \param decoder the new decoder that should be used instead of the current one.

    This method allows to change the text decoder runtime.
    
    \note The TextReader is never owning the transcoder.
    */
   void setEncoding( Transcoder* decoder );
   
   Transcoder* transcoder() const { return m_decoder; }

   /** Reads a text from the stream.
    \param str A target string where to put the data.
    \param count The count of characters that must be read from the stream.
    \param start Start position where to put the decoded content in the target string.
    \return True if a string could be read.
    \throw EncodingError if the input stream has encoding errors.

    This method reads at maximum count characters (the actual count of read bytes
    may differ depending on the encoding, and stores them in the target Falcon string.

    If the string is not large enough to read count character, it is resized accordingly.
    If less than count characters can be read from the stream, the resulting string
    will be smaller, but the memory wont be automatically trimmed down to its size.
    */
   bool read( String& str, length_t count, length_t start = 0 );


   /** Reads a text from the stream.
    \param str A target string where to put the data.
    \return True if a string could be read.
    \throw EncodingError if the input stream has encoding errors.

    This method fills the string to it's full pre-allocated size (if enough
    characters can be read from the stream).
    */
   bool read( String& str );

   /** Reads all the remaining tesxt from the stream.
    \param str A target string where to put the data.
    \return True if a string could be read.
    \throw EncodingError if the input stream has encoding errors.

    This method fills a string with all the contents of a text file.

    The algorithm stores the target text in smaller pre-allocated strings,
    merging them after the full size and character width of the final string
    is known. This makes relatively efficient to read very large text files, but
    it requires an extra amount of temporary memory to perform the operation.
    */
   bool readEof( String& str );

   /** Reads the next text line from a stream.
    \param str A target string where to put the data.
    \param maxCount The maximum count of characters to read before giving off.
    \param start Start from a determined position in the string.
    \return True if a line could be read, false if EOF was hit or maxCount
            characters had been read before hitting and end-of-line.
    \throw EncodingError if the input stream has encoding errors.

    This method scans a stream in search for a LF, CRLF or LFCR sequence
    (whichever comes first). If maxCount characters are read before finding
    such a sequence, the method returns false, but the read contents are stored
    in the target string.

    The string is re-allocated and resized as needed.

    The end-of-line sequence is NOT returned in the target string.

    To loop on all the lines of the lines in a file, the following strategy
    can be adopted:

    \code
    String tgt;
    Reader reader( ... );

    while( true ) {
      if ( ! reader.readLine( tgt, 1024 ) )
      {
         // either we're eof, or the line was longer than 1024 characters.
         if ( tgt.size() == 0 ) {
            break; // we're eof
         }

         // manage unterminated line, possibly raise an error
      }
      else
      {

         // manage a line
      }
    }
    \endcode    
    */
   bool readLine( String& target, length_t maxCount );

   /** Reads a text file up to a certain separator.
    \param str A target string where to put the data.
    \param separator A separator used to terminate the record.
    \param maxCount The maximum count of characters to read before giving off.
    \param start Start from a determined position in the string.
    \return True if a record could be read, false if EOF was hit or maxCount
            characters had been read before hitting the separator.
    \throw EncodingError if the input stream has encoding errors.
    \see readLine

    This method is similar to readLine, but it is possible indicate a string that
    will be used as field separator.
    */
   bool readRecord( String& target, const String& separator, length_t maxCount );
   
   /** Reads a text from a text file, using multiple separators as terminators.
    \param str A target string where to put the data.
    \param tokens An array of Falcon strings.
    \param tokenCount Number of tokens in the array.
    \param maxCount The maximum count of characters to read before giving off.
    \param start Start from a determined position in the string.
    \return The number of the token hit, or -1 if EOF was hit or maxCount
            characters had been read before hitting one of the separators.
    \throw EncodingError if the input stream has encoding errors.
    \see readLine
    
    This method is similar to readLine, but it is possible indicate a multiple
    strings that could be considere the limit of the read
    */
   int readToken( String& target, const String* tokens, int32 tokenCount, length_t maxCount );

   /** Reads exactly one character.
    \return A character value or NoChar (-1 cast to char_t)  if hit EOF .
    \throw EncodingError if the input stream has encoding errors.

    Returns a single character read from the stream and decoded by the transcoder.

    Notice that using this method to parse text streams may be extremely
    unefficient.
    */
   char_t getChar();


   /** Puts a single character back on the reader.
    \param chr The character to be pushed back.
    \throw EncodingError if the input stream has encoding errors.

    This method puts back a single character that will be read back when
    the next read method is called. The character will be appended to the
    target before starting the decoding of the data coming to the stream,
    or will be returned by getChar() if that is called first.

    However, the pushed back character won't be taken into consideration when
    scanning for a token in the stream.

    Calling this method multiple times changes the previously pushed character;
    in other words, it's not possible to unget mor than one character at a time.
    */
   void ungetChar( char_t chr );

   Stream* underlying() const { return m_stream; }

   void gcMark( uint32 mark ) { m_mark = mark; }
   uint32 currentMark() const { return m_mark; }

   virtual bool eof() const;

protected:
   char_t m_pushedChr;
   Transcoder* m_decoder;

   /** Structure holding the standard new-line tokens */
   struct std_token {
      byte data[12];
      length_t size;
   };
   
   struct std_token m_nl[3];

   struct token {
      byte* data;
      length_t size;
   };

   struct token* m_cTokens;
   int m_cTSize;

   int readTokenInternal( String& target, struct token* tokens, int32 tokenCount, length_t maxCount );
   
   /** Finds the first token available in the buffer. */
   int findFirstToken( struct token* toks, int tcount, length_t& pos );

   void clearTokens();

   void makeDefaultSeps();
private:
   uint32 m_mark;
};

}

#endif	/* _FALCON_TEXTREADER_H */

/* end of textreader.h */
