/*
   FALCON - The Falcon Programming Language.
   FILE: bufferedstream.h
   $Id: bufferedstream.h,v 1.1.1.1 2006/10/08 15:05:38 gian Exp $

   Buffered stream
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ago 18 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
   byte *m_buffer;
   uint64 m_filePos;
   int32 m_bufPos;
   int32 m_bufLen;
   int32 m_bufSize;
   bool m_changed;

   bool refill();

protected:
   virtual int64 seek( int64 pos, e_whence whence );
public:
   BufferedStream( uint32 bufSize = default_buffer_size );
   virtual ~BufferedStream();

   virtual int32 read( byte *buffer, int32 size );
   virtual int32 write( const byte *buffer, int32 size );
   virtual int64 tell();
   virtual bool truncate( int64 pos = - 1 );

   virtual int32 readAvailable( int32 msecs_timeout );
   virtual int32 writeAvailable( int32 msecs_timeout );

   virtual bool writeString( const String &content, uint32 begin = 0, uint32 end = csh::npos );
   virtual bool readString( String &content, uint32 size );

   virtual bool close();

   bool flush();
};

}

#endif

/* end of bufferedstream.h */
