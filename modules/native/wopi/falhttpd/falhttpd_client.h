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

namespace Falcon
{

class FalhttpdClient
{
public:
   FalhttpdClient( const FalhttpOptions& opts, Mod::Socket* skt );
   ~FalhttpdClient();

   void close();
   void serve();

   void replyError( int errorID, const String& explain="" );
   String codeDesc( int errorID );
   String getServerSignature();

   void sendData( const String& sReply );
   void sendData( const void* data, uint32 size );

   const FalhttpOptions& options() const { return m_options; }
   Mod::Socket* skt() const { return m_skt; }
   Stream* stream() const { return m_stream; }
   WOPI::Reply* reply() const { return m_reply; }

private:
   void serveRequest(
         const String& sMethod, const String& sUri,  const String& sProto, Stream* si );

   Mod::Socket* m_skt;
   bool m_bComplete;
   Stream* m_stream;
   WOPI::Reply* m_reply;

   const FalhttpOptions& m_options;
};

}

#endif /* FALHTTPD_CLIENT_H_ */

/* falhttpd_client.h */
