/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_istream.h

   Micro HTTPD server providing Falcon scripts on the web.

   Servicing single client.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 28 Feb 2010 23:03:18 +0100
s
   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALHTTPD_ISTREAM_H_
#define FALHTTPD_ISTREAM_H_

#include "falhttpd.h"

#include <falcon/stream.h>

class SocketInputStream: public Falcon::Stream
{
public:
   SocketInputStream( SOCKET s );
   virtual ~SocketInputStream();

   virtual SocketInputStream* clone() const;
   virtual Falcon::int32 read( void *buffer, Falcon::int32 size );
   virtual bool get( Falcon::uint32 &chr );
   virtual Falcon::int64 lastError() const;

private:

   SOCKET m_socket;
   Falcon::int64 m_nLastError;
};

#endif

/* falhttpd_istream.h */
