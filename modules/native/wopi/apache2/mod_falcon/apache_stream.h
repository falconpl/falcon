/*
   FALCON - The Falcon Programming Language.
   FILE: apache_stream.h

   Falcon module for Apache 2
   Apache specific Falcon Stream.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-07-28 18:55:17

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef APACHE_STREAM_H
#define APACHE_STREAM_H

#include <falcon/stream.h>

class ApacheOutput;

// Class translating falcon output requests to apache writes.
class ApacheStream: public Falcon::Stream
{
   // The owning VM. Needed as the first write must send back
   // the request.
   ApacheOutput *m_output;
   Falcon::int64 m_lastError;

public:
   ApacheStream( ApacheOutput *aout );
   ApacheStream( const ApacheStream &other );

    // We don't really need to implement all of those;
   // as we want to reimplement output streams, we'll just
   // set "unsupported" where we don't want to provide support.
   bool writeString( const Falcon::String &source, Falcon::uint32 begin=0, Falcon::uint32 end = Falcon::csh::npos );
   virtual bool close();
   virtual Falcon::int32 write( const void *buffer, Falcon::int32 size );
   virtual Falcon::int32 writeAvailable( int, const Falcon::Sys::SystemData* );
   virtual Falcon::int64 lastError() const;
   virtual bool put( Falcon::uint32 chr );
   virtual bool get( Falcon::uint32 &chr );
   virtual ApacheStream *clone() const;

   // Flushes the stream.
   virtual bool flush();
};

// Class translating falcon output requests to apache reads.
class ApacheInputStream: public Falcon::Stream
{
public:
   ApacheInputStream(  request_rec *request );
   virtual ~ApacheInputStream();

   virtual bool close();
   virtual Falcon::int32 read( void *buffer, Falcon::int32 size );
   virtual Falcon::int32 readAvailable( int, const Falcon::Sys::SystemData* );
   virtual Falcon::int64 lastError() const;
   virtual bool get( Falcon::uint32 &chr );
   virtual ApacheInputStream *clone() const;

private:
   Falcon::int64 m_lastError;
   request_rec* m_request;
   apr_bucket_brigade *m_brigade;
};


#endif

/* end of apache_stream.h */

