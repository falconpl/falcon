/*
   FALCON - The Falcon Programming Language.
   FILE: bufferedstream.h

   Buffered stream
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ago 18 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Buffered stream.
*/

#ifndef flc_bufferedstream_H
#define flc_bufferedstream_H

#include <falcon/fstream.h>

namespace Falcon {

/** Buffered version of basic stream.
   \TODO Everything except reading straight is to be tested.
*/
class FALCON_DYN_CLASS BufferedStream: public FileStream
{
public:
	enum {
		default_buffer_size = 1024
	} enum_default_buffer_size ;

private:
   int32 m_bufSize;
   bool m_changed;
   byte *m_buffer;
   int32 m_bufPos;
   int32 m_bufLen;
   uint64 m_filePos;

   bool refill();

protected:
   virtual int64 seek( int64 pos, e_whence whence );
public:
   BufferedStream( uint32 bufSize = default_buffer_size );
   virtual ~BufferedStream();

   virtual int32 read( void *buffer, int32 size );
   virtual int32 write( const void *buffer, int32 size );
   virtual int64 tell();
   virtual bool truncate( int64 pos = -1 );

   virtual int32 readAvailable( int32 msecs_timeout, const Sys::SystemData *data = 0 );
   virtual int32 writeAvailable( int32 msecs_timeout, const Sys::SystemData *data = 0 );

   virtual bool writeString( const String &content, uint32 begin = 0, uint32 end = csh::npos );
   virtual bool readString( String &content, uint32 size );

   virtual bool close();

   bool flush();
};

}

#endif

/* end of bufferedstream.h */
