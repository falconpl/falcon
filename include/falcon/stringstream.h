/*
   FALCON - The Falcon Programming Language.
   FILE: stringstream.h

   Straem for stream-like I/O to memory.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Apr 2011 21:57:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_STRINGSTREAM_H
#define _FALCON_STRINGSTREAM_H

#include <falcon/string.h>
#include <falcon/stream.h>
#include <falcon/multiplex.h>
#include <falcon/mt.h>
#include <falcon/syncqueue.h>


namespace Falcon {

class StdMpxFactories;
class Selectable;

class FALCON_DYN_CLASS StringStream: public Stream
{
public:
   StringStream( int32 size=0 );
   StringStream( byte* data, int64 size );
   StringStream( const String &strbuf );
   StringStream( const StringStream &strbuf );
   virtual ~StringStream();

   virtual bool close();
   virtual size_t read( void *buffer, size_t size );
   virtual size_t write( const void *buffer, size_t size );

   virtual bool setNonblocking( bool ) { return true; }
   virtual bool isNonbloking() const { return true; }

   virtual int64 tell();
   virtual bool truncate( int64 pos=-1 );

   uint32 length() const;
   uint32 allocated() const;
   byte *data() const;

   virtual const Class* handler() const;

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
   
   /** Gets a string containing the content of the stream and empties the stream
	  \return a string containing all the data in the stream (may be empty).
   */
   String getStringAndClear();

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

   /** Changes the pipe mode of this string stream.
    *
    * In pipe mode, the string stream read and write pointers are different.
    *
    * If the mode is set to false, they move together, and the write pointer
    * is reset to the position of the read pointer.
    *
    * In pipe mode, seek() moves both the read and the write pointer,
    * and current position is relative to write pointer,
    * but tell() returns the read pointer.
    */
   void setPipeMode( bool mode );

   /** Changes the pipe mode of this string stream.
    *
    * In pipe mode, the string stream read and write pointers are different.
    */
   bool isPipeMode() const ;
  
   virtual const Multiplex::Factory* multiplexFactory() const;

protected:
   int64 m_posRead;
   int64 m_posWrite;
   bool m_bPipeMode;

   virtual int64 seek( int64 pos, e_whence whence );

   void setBuffer( const String &source );
   void setBuffer( const char* source, int size=-1 );
   bool detachBuffer();

   bool subWriteString( const String &source );

private:
   class Buffer;
   Buffer* m_b;
   Selectable* m_selectable;

   class MpxFactory: public Multiplex::Factory
   {
   public:
      MpxFactory() {}
      virtual ~MpxFactory();
      virtual Multiplex* create( Selector* selector ) const;
   };

   friend class StdMpxFactories;

   class MPX;
   friend class MPX;
   friend class Traits;
};

}

#endif

/* end of stringstream.h */
