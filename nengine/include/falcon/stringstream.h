/*
   FALCON - The Falcon Programming Language.
   FILE: file_string.h

   Management of membuffer strings; directly included by file_base.h
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 13 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Management of membuffer strings; directly included by file_base.h.
*/

#ifndef _FALCON_STRINGSTREAM_H
#define _FALCON_STRINGSTREAM_H

#include <falcon/string.h>
#include <falcon/stream.h>

namespace Falcon {

class FALCON_DYN_CLASS StringStream: public Stream
{
public:
   StringStream( int32 size=0 );
   StringStream( const String &strbuf );
   StringStream( const StringStream &strbuf );
   virtual ~StringStream();

   virtual bool close();
   virtual size_t read( void *buffer, size_t size );
   virtual size_t write( const void *buffer, size_t size );
   virtual size_t readAvailable( int32 msecs_timeout=0 );
   virtual size_t writeAvailable( int32 msecs_timeout=0 );

   virtual int64 tell();
   virtual bool truncate( int64 pos=-1 );

   uint32 length() const;
   uint32 allocated() const;
   byte *data() const;

   /** Transfers a string stream buffer into this one.
      The original buffer is emptied, and this buffer aqcuires the
      same status, allocation and contents of the other one.
   */
   void transferFrom( StringStream &strbuf );

   /** Gets a string copying the content of the stream.
      The memory that is currently held in this object is copied in a string.
      Read-write operations can then continue, and the status of the object
      is not changed.
      \param target The string where the buffer is copied.
   */
   void getString( String &target ) const;
   
   /** Gets a string copying the content of the stream, newly allocating the target string.
      The memory that is currently held in this object is copied in a string.
      Read-write operations can then continue, and the status of the object
      is not changed.
      \return a string containing all the data in the stream (may be empty, but not 0).
   */
   String *getString() const
   {
      String *temp = new String;
      getString( *temp );
      return temp;
   }
   

   /** Gets the phisical memory created by this object and turns it into a string.
      The memory that has been created by the stream-like operations is directly
      passed into a string object, in a very efficient way; as a result, the buffer
      in this object is transferred as-is into the returned string, and this object
      becomes unuseable (closed).

      If the stream has already been closed, the function will return 0.

      \return a string containing all the data in the stream.
   */
   String *closeToString();
   
   
   /** Gets the phisical memory created by this object and turns it into a string.
      This version of the method stores the phisical memory in the given string,
      and configures it as a single byte memory buffer string.

      \return false if the stream has already been closed.
   */
   bool closeToString( String &target );

   /** Gets the phisical memory created by this object.
      This version of the method retreives the internally allocated buffer and
      empties this StringStream.

      The returned buffer must be de-allocated using Falcon::memFree()

      \return a byte * that will receive the internally created data.
   */
   byte  *closeToBuffer();
  
   virtual StringStream *clone() const;
  
protected:
   uint32 m_pos;
   virtual int64 seek( int64 pos, e_whence whence );

   void setBuffer( const String &source );
   void setBuffer( const char* source, int size=-1 );
   bool detachBuffer();

   bool subWriteString( const String &source );

private:
   class Buffer;
   Buffer* m_b;
};

}

#endif

/* end of file_string.h */
