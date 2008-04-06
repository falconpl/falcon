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
protected:
   byte *m_membuf;
   uint32 m_length;
   uint32 m_allocated;
   uint32 m_pos;
   int32 m_lastError;

   virtual int64 seek( int64 pos, e_whence whence );
public:
   StringStream( int32 size=0 );
   StringStream( const String &strbuf );
   StringStream( const StringStream &strbuf );

   virtual ~StringStream() { close(); }

   virtual bool close();
   virtual int32 read( void *buffer, int32 size );
   virtual bool readString( String &dest, uint32 size );
   virtual int32 write( const void *buffer, int32 size );
   virtual bool writeString( const String &source, uint32 begin = 0, uint32 end = csh::npos );
   virtual bool put( uint32 chr );
   virtual bool get( uint32 &chr );
   virtual int32 readAvailable( int32 msecs );
   virtual int32 writeAvailable( int32 msecs );

   virtual int64 tell();
   virtual bool truncate( int64 pos=-1 );

   uint32 length() const { return m_length; }
   uint32 allocated() const { return m_allocated; }
   byte *data() const { return m_membuf; }

   virtual bool errorDescription( ::Falcon::String &description ) const;

   /** Gets a string copying the content of the stream.
      The memory that is currently held in this object is copied in a string.
      Read-write operations can then continue, and the status of the object
      is not changed.
      \return a string containing all the data in the stream.
   */
   String *getString() const;

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

   virtual int64 lastError() const { return (int64) m_lastError; }

   virtual UserData *clone();
};

}

#endif

/* end of file_string.h */
