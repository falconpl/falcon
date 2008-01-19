/*
 * FALCON - The Falcon Programming Language.
 * FILE: zlibstream.h
 *
 * Stream to and from zlib
 * -------------------------------------------------------------------
 * (c) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 * In order to use this file in its compiled form, this source or
 * part of it you have to read, understand and accept the conditions
 * that are stated in the LICENSE file that comes boundled with this
 * package.
 */

/** \file
 * Stream to and from zlib
 */

#ifndef flc_zlibstream_h
#define flc_zlibstream_h

#include <falcon/string.h>
#include <falcon/stream.h>

#include "zlib.h"

namespace Falcon {

class FALCON_DYN_CLASS ZLibStream: public Stream
{
protected:
   int64 m_lastError;

   virtual int64 seek( int64 pos, e_whence whence );

public:
   ZLibStream( int32 size=0 );
   ZLibStream( const ZLibStream &zlibstream );

   virtual ~ZLibStream();

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

   //uint32 length() const { return m_length; }
   //uint32 allocated() const { return m_allocated; }
   //byte *data() const { return m_membuf; }

   virtual bool errorDescription( ::Falcon::String &description ) const;
   virtual int64 lastError(void) const;

   virtual UserData *clone();
};
}


#endif /* flc_zlibstream_h */

/* end of zlibstream.h */
