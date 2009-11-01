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

#ifndef flc_file_string_H
#define flc_file_string_H

#include <falcon/string.h>
#include <falcon/stream.h>

namespace Falcon {

class FALCON_DYN_CLASS StringStream: public Stream
{
private:
   class Buffer;
   Buffer* m_b;
   
   
protected:
   uint32 m_pos;
   virtual int64 seek( int64 pos, e_whence whence );
   
   void setBuffer( const String &source );
   void setBuffer( const char* source, int size=-1 );
   bool detachBuffer();
   
public:
   StringStream( int32 size=0 );
   StringStream( const String &strbuf );
   StringStream( const StringStream &strbuf );

   virtual ~StringStream();

   virtual bool close();
   virtual int32 read( void *buffer, int32 size );
   virtual bool readString( String &dest, uint32 size );
   virtual int32 write( const void *buffer, int32 size );
   virtual bool writeString( const String &source, uint32 begin = 0, uint32 end = csh::npos );
   virtual bool put( uint32 chr );
   virtual bool get( uint32 &chr );
   virtual int32 readAvailable( int32 msecs, const Sys::SystemData *sysData = 0 );
   virtual int32 writeAvailable( int32 msecs, const Sys::SystemData *sysData = 0 );

   virtual int64 tell();
   virtual bool truncate( int64 pos=-1 );

   uint32 length() const;
   uint32 allocated() const;
   byte *data() const;

   virtual bool errorDescription( ::Falcon::String &description ) const;

   /** Transfers a string stream buffer into this one.
      The original buffer is emptied, and this buffer aqcuires the
      same status, allocation and contents of the other one.
   */
   void transfer( StringStream &strbuf );

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
   
   /** Gets a string copying the content of the stream, allocating in the garbage the target string.
      The memory that is currently held in this object is copied in a string.
      Read-write operations can then continue, and the status of the object
      is not changed.
      \return a string containing all the data in the stream (may be empty, but not 0).
   */
   CoreString *getCoreString() const
   {
      CoreString *temp = new CoreString;
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
   
   /** Gets the phisical memory created by this object and turns it into a newly created garbage collected string.
      \see closeToString()
      \return a string containing all the data in the stream.
   */
   CoreString *closeToCoreString();
   
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

   virtual int64 lastError() const;
   virtual StringStream *clone() const;
   virtual void gcMark( uint32 mark ) {}
};

}

#endif

/* end of file_string.h */
