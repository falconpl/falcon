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
#include <falcon/wopi/request.h>
#include <falcon/wopi/mem_sm.h>

#define DEFAULT_BUFFER_SIZE 4096

using namespace Falcon;

class FalhttpdClient
{
public:
   FalhttpdClient( const FalhttpOptions& opts, LogArea* l, SOCKET s, const String& remName );
   ~FalhttpdClient();

   void close();
   void serve();

   void replyError( int errorID, const String& explain="" );
   String codeDesc( int errorID );
   String getServerSignature();

   void sendData( const String& sReply );
   void sendData( const void* data, uint32 size );

   const FalhttpOptions& options() const { return m_options; }
   Falcon::LogArea* log() const { return m_log; }
   SOCKET socket() const { return m_nSocket; }
   Falcon::WOPI::SessionManager* smgr() const { return m_pSessionManager; }

private:
   void serveRequest(
         const String& sRequest, const String& sUri,  const String& sProto,
         Stream* si );

   LogArea* m_log;
   SOCKET m_nSocket;
   String m_sRemote;

   char* m_cBuffer;
   uint32 m_sSize;
   bool m_bComplete;

   const FalhttpOptions& m_options;
   Falcon::WOPI::SessionManager* m_pSessionManager;
};

#endif /* FALHTTPD_CLIENT_H_ */

/* falhttpd_client.h */
