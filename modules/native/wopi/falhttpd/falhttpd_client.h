/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_client.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   Servicing single client.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 24 Feb 2010 20:10:45 +0100
s
   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALHTTPD_CLIENT_H_
#define FALHTTPD_CLIENT_H_

#include "falhttpd.h"
#include <inet_mod.h>
#include <falcon/wopi/request.h>
#include <falcon/wopi/reply.h>
#include <falcon/wopi/mem_sm.h>
#include <falcon/wopi/client.h>
#include <falcon/wopi/errorhandler.h>

namespace Falcon
{

class FalhttpdClient: public WOPI::Client
{
public:
   FalhttpdClient( const FalhttpOptions& opts, Mod::Socket* skt );
   virtual ~FalhttpdClient();

   void close();
   void serve();

   const FalhttpOptions& options() const { return m_options; }
   Mod::Socket* skt() const { return m_skt; }

   WOPI::ErrorHandler& errhand() { return m_errhand; }

   void replyError( int code, const String& msg = "" ) { m_errhand.replyError(this, code, msg); }
private:
   void serveRequest(
         const String& sMethod, const String& sUri,  const String& sProto );

   Mod::Socket* m_skt;
   const FalhttpOptions& m_options;

   class SHErrorHandler: public WOPI::ErrorHandler
   {
   public:
      SHErrorHandler() {}
      virtual ~SHErrorHandler() {}
      virtual void replyError( WOPI::Client* client, int code, const String& message );
   }
   m_errhand;
};

}

#endif /* FALHTTPD_CLIENT_H_ */

/* falhttpd_client.h */
