/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_ostream.h

   Micro HTTPD server providing Falcon scripts on the web.

   Stream reading input from a TCP socket.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 06 Mar 2010 20:48:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALHTTPD_OSTREAM_H_
#define FALHTTPD_OSTREAM_H_

#include "falhttpd.h"

#include <falcon/stream.h>
class FalhttpdReply;

class FalhttpdOutputStream: public Falcon::Stream
{
public:
   FalhttpdOutputStream( SOCKET s );
   virtual ~FalhttpdOutputStream();

   virtual FalhttpdOutputStream* clone() const;
   virtual Falcon::int32 write( const void *buffer, Falcon::int32 size );
   virtual bool put( Falcon::uint32 chr );
   virtual Falcon::int64 lastError() const;
   virtual bool get( Falcon::uint32 &chr );


private:

   SOCKET m_socket;
   Falcon::int64 m_nLastError;
};

#endif

/* falhttpd_ostream.h */
