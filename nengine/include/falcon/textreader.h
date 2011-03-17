/*
   FALCON - The Falcon Programming Language.
   FILE: textreader.cpp

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
class TextDecoder;

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

 */
class FALCON_DYN_CLASS TextReader: public Reader
{
public:
   TextReader( Stream* stream, const String& encoding, bool bOwn = false );
   virtual ~TextReader();

   void setEncoding( const String& encoding );

   bool read( String& str, length_t count, length_t start = 0 );
   bool read( String& str );

   bool readline( String& target, length_t maxCount, length_t start = 0 );
   bool readRecord( String& target, const String& separator, length_t maxCount, length_t start = 0 );
   bool readToken( String& target, const String* tokens, int32 tokenCount, length_t maxCount, length_t start = 0 );

   int32 getChar();
   void ungetChar( int32 chr );

private:
   int32 m_pushedChr;
   bool m_bOwnStream;

   Stream* m_stream;
   TextDecoder* m_decoder;

   String m_recordBuffer;
   length_t m_nRecordPos;
};

}

#endif	/* _FALCON_TEXTREADER_H */

/* end of textreader.h */
