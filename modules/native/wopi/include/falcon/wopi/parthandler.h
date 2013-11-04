/*
   FALCON - The Falcon Programming Language.
   FILE: parthandler.h

   Web Oriented Programming Interface

   RFC1867 compliant upload handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 26 Feb 2010 20:29:47 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_WOPI_PARTHANDLER_H_
#define _FALCON_WOPI_PARTHANDLER_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/stream.h>

#include <deque>
#include <map>


namespace Falcon {

class StringStream;
class MemBuf;

namespace WOPI {

class Request;

class HeaderValue
{
public:
   typedef std::deque<String> ValueList;
   typedef std::map<String, String> ParamMap;

   HeaderValue();
   HeaderValue( const String& value );
   HeaderValue( const HeaderValue& other );

   virtual ~HeaderValue();

   bool hasValue() const { return ! m_lValues.empty(); }
   const String& value() const { return m_lValues.front(); }
   const ValueList& values() const { return m_lValues; }
   ValueList& values() { return m_lValues; }

   const ParamMap& parameters() const { return m_mParameters; }
   ParamMap& parameters() { return m_mParameters; }

   bool parseValue( const String& v );

   const String& rawValue() const { return m_sRawValue; }

   void setRawValue( const String& value );

private:
   bool parseValuePart( const String& v );
   bool parseParamPart( const String& v );

   ValueList m_lValues;
   ParamMap m_mParameters;
   String m_sRawValue;
};

/** RFC1867 compliant upload handler.
*/
class PartHandler
{
public:
   typedef std::map<String, HeaderValue> HeaderMap;


   PartHandler();

   PartHandler( const String& sBoundary );

   virtual ~PartHandler();

   void parse( Stream* input )
   {
      bool dummy;
      parse( input, dummy );
   }

   void parse( Stream* input, bool& isLast );

   //! Return true if this part is complete
   bool isComplete() const { return m_nReceived == m_nPartSize; }

   //! Returns the declared content lenght of this part.
   int64 contentLenght() const { return m_nPartSize; }

   //! Returns the name of the part.
   const String& name() const { return m_sPartName; }

   //! Returns the original filename of an uploaded file
   const String& filename() const { return m_sPartFileName; }

   //! Returns the content type of this section
   const String& contentType() const { return m_sPartContentType; }

   //! Returns the temporary file where this part has been saved.
   const String& tempFile() const { return m_sTempFile; }

   PartHandler* next() const { return m_pNextPart; }
   PartHandler* child() const { return m_pSubPart; }

   //! Adds an header
   void addHeader( const String& key, const String& value );

   //! Returns true if we're saving on a memory stream.
   bool isMemory() const { return m_str_stream != 0; }

   //! Parses just the header before the body
   void parseHeader( Stream* input );

   /** Parse the body of the request part.
    \note The upload stream MUST have been already opened or the function will assert.
    \param input The stream.
    \param isLast will be set to true if this is the last part of a multipart request.
    */
   void parseBody( Stream* input, bool& isLast );

   /* Returns the memory data in a target string.

      If the part was loaded into memory, the memory data is directly transfered to this string
      without any copy.

      Otherwise, nothing is done and the function returns false.
   */
   bool getMemoryData( String& target );

   const HeaderMap& headers() const { return m_mHeaders; }
   HeaderMap& headers() { return m_mHeaders; }

   int64 uploadedSize() const { return m_nReceived; }

   void startUpload();
   void closeUpload();
   void startMemoryUpload();
   void startFileUpload();

   void setOwner( Request* owner ) { m_owner = owner; }

   /** Sets the boundary from outside.
       Useful if the header are parsed elsewhere.
   */
   void setBoundary( const String& b ) { m_sBoundary = b; }

   //! Create a new buffer (that will be passed down to other elements in the chain)
   void initBuffer();

   bool uploadsInMemory() const { return m_bUseMemoryUpload; }

   //! Activates memory storage of uploaded data.
   void uploadsInMemory( bool b ) { m_bUseMemoryUpload = b; }

   //! True if this part is meant to upload a file, false if it's a standard field.
   bool isFile() const { return m_bPartIsFile; }

   void setBodySize( int64 size ) { m_nBodyLeft = size; }

private:

   class PartHandlerBuffer
   {
   public:
      enum e_constants {
         buffer_size = 8192
      };

      PartHandlerBuffer( int64* pToMax );
      ~PartHandlerBuffer();

      uint32 m_nBufPos;
      uint32 m_nBufSize;
      byte* m_buffer;

      bool full() { return m_nBufSize == buffer_size; }
      bool empty() { return m_nBufSize == 0; }

      //! Flushes 0->bufPos, and moves m_nBufPos->m_nBufSize -> 0
      void flush( Stream* out = 0 );

      //! Flushes the size (all)
      void allFlush( Stream* out = 0 )
      {
         m_nBufPos = m_nBufSize;
         flush(out);
      }

      //! Searches this string in the buffer.
      uint32 find( const String& str );

      //! refills buffer
      bool fill( Stream* input );

      //! gets a string from the buffer (without allocating it).
      void grab( String& target, int posEnd );

      void grabMore( String& target, int size );

      bool hasMore( int count, Stream* input, Stream* output = 0 );

   private:

      int64* m_nDataLeft;
   };

   PartHandlerBuffer* m_pBuffer;
   bool m_bOwnBuffer;

   void passSetting( PartHandler* child );


   //! Searches for the boundary and store the data in m_stream
   void scanForBound( const String& boundary, Stream* input, bool& isLast );


   void parseHeaderField( const String& line );

   void parsePrologue( Stream* input );
   void parseMultipartBody( Stream* input );

   HeaderMap m_mHeaders;

   String m_sPartName;
   String m_sPartFileName;
   String m_sPartContentType;
   String m_sTempFile;

   String m_sBoundaryBuffer;
   String m_sBoundary;
   String m_sEnclosingBoundary;

   int64 m_nPartSize;
   // Pointer to the total size expected to be received.
   int64* m_pToBodyLeft;
   int64 m_nBodyLeft;

   int64 m_nReceived;
   Stream* m_stream;
   // for memory uploads and data fields
   StringStream* m_str_stream;

   PartHandler *m_pNextPart;
   PartHandler *m_pSubPart;

   Request* m_owner;
   bool m_bPartIsFile;
   bool m_bUseMemoryUpload;
};

}
}

#endif /* PARTHANDLER_H_ */

/* end of parthandler.h */

