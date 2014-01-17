/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_eh.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   Error handler for falhttpd.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 24 Feb 2010 20:10:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/autocstring.h>

#include "falhttpd_eh.h"
#include "falhttpd_client.h"

namespace Falcon {

SHErrorHandler::SHErrorHandler( bool bFancy ):
    ErrorHandler(bFancy)
{
}

SHErrorHandler::~SHErrorHandler()
{
}


void SHErrorHandler::replyError( WOPI::Client* client, const String& message )
{
   AutoCString content(message);
   client->sendData( content.c_str(), content.length() );
}


void SHErrorHandler::logSysError( WOPI::Client* client, int code, const String& message )
{
   FalhttpdClient* cli = static_cast<FalhttpdClient*>(client);
   LOGE( String("System Error ").N(code).A(" ").A(message).A(" to ").A( cli->skt()->address()->toString()) );
}


void SHErrorHandler::logError( WOPI::Client* client, Error* error )
{
   String head;
   FalhttpdClient* cli = static_cast<FalhttpdClient*>(client);
   LOGE( String("Script Error ").A(error->heading(head)).A(" to ").A(cli->skt()->address()->toString()) );
}

}

/* falhttpd_eh.cpp */
