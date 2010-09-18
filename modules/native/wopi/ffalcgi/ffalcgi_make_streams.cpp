/*
   FALCON - The Falcon Programming Language.
   FILE: falcgi_make_streams.cpp

   Falcon CGI program driver - CGI specific stream provider.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Feb 2010 17:35:14 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <cgi_make_streams.h>
#include <falcon/stream.h>

#include <falcon/stdstreams.h>

#include <errno.h>

#include <fastcgi.h>
#include <fcgi_stdio.h>

class FFCGIStream: public Falcon::Stream
{
public:
   FFCGIStream( FILE* tgt );

   virtual ~FFCGIStream();

   virtual Falcon::int32 write( const void* data, Falcon::int32 size );
   virtual Falcon::int32 read( void* data, Falcon::int32 size );
   virtual bool put( Falcon::uint32 chr );
   virtual bool get( Falcon::uint32& chr );
   virtual bool close();
   virtual Falcon::int64 lastError() const;
   virtual bool flush();

private:
   FILE* m_target;
};

FFCGIStream::FFCGIStream( FILE* tgt ):
      Stream(t_stream),
      m_target(tgt)
{

}

FFCGIStream::~FFCGIStream()
{
}


Falcon::int32 FFCGIStream::write( const void* data, Falcon::int32 size )
{
   return fwrite( const_cast<void*>(data), 1, size, m_target );
}

Falcon::int32 FFCGIStream::read( void* data, Falcon::int32 size )
{
   return fread( data, 1, size, m_target );
}

bool FFCGIStream::put( Falcon::uint32 chr )
{
   return fputc( chr, m_target ) >= 0 ;
}

bool FFCGIStream::get( Falcon::uint32& chr )
{
   int ichr = fgetc( m_target );
   chr = (Falcon::uint32) ichr;
   return ichr >= 0;
}

bool FFCGIStream::close()
{
   // we can't close this streams
   return true;
}

Falcon::int64 FFCGIStream::lastError() const
{
   return (Falcon::int64) errno;
}


bool FFCGIStream::flush()
{
   return fflush( m_target ) == 0 ;
}

Falcon::Stream* makeOutputStream()
{
   Falcon::Stream* test = new  FFCGIStream( stdout );
   return test;
}

Falcon::Stream* makeInputStream()
{
   return new FFCGIStream( stdin );
}

Falcon::Stream* makeErrorStream()
{
   return new FFCGIStream( stderr );
}

/* end of falcgi_make_streams.cpp */
