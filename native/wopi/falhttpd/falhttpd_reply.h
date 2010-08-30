/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_reply.h

   Micro HTTPD server providing Falcon scripts on the web.

   Reply object specific for the Falcon micro HTTPD server.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 06 Mar 2010 21:31:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALHTTPD_REPLY_H_
#define FALHTTPD_REPLY_H_

#include "falhttpd.h"
#include <falcon/wopi/reply.h>

class FalhttpdReply: public Falcon::WOPI::Reply
{
public:
   FalhttpdReply( const Falcon::CoreClass* cls );
   ~FalhttpdReply();
   void init( SOCKET s );

   Falcon::Stream* makeOutputStream();
   virtual void startCommit();
   virtual void commitHeader( const Falcon::String& hname, const Falcon::String& hvalue );
   virtual void endCommit();

   static Falcon::CoreObject* factory( const Falcon::CoreClass* cls, void* ud, bool bDeser );

private:
   void send( const Falcon::String& s );
   SOCKET m_socket;

   Falcon::String m_headers;
};

#endif

/* falhttpd_reply.h */
