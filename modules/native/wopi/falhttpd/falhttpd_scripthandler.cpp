/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_scripthandler.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   Request handler processing a script.
   Mimetype is not determined (must be set by the script), but it
   defaults to text/html; charset=utf-8

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Mar 2010 12:18:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "falhttpd_scripthandler.h"
#include "falhttpd_client.h"
#include "falhttpd.h"

#include <falcon/falcon.h>
#include <falcon/sys.h>
#include <falcon/wopi/modulewopi.h>
#include <falcon/wopi/replystream.h>
#include <falcon/wopi/reply.h>
#include <falcon/wopi/classrequest.h>
#include <falcon/wopi/stream_ch.h>

namespace Falcon {

ScriptHandler::ScriptHandler( const Falcon::String& sFile, FalhttpdClient* cli ):
      FalhttpdRequestHandler( sFile, cli )
{
   m_runner = new WOPI::ScriptRunner( "FalHTTPD", FalhttpdApp::get()->vm(), FalhttpdApp::get()->log(), FalhttpdApp::get()->eh() );
}


ScriptHandler::~ScriptHandler()
{
   delete m_runner;
}


void ScriptHandler::serve()
{
   // TODO: configure directly at startup.
   m_runner->textEncoding(m_client->options().m_sTextEncoding);
   m_runner->sourceEncoding(m_client->options().m_sSourceEncoding);
   m_runner->loadPath(m_client->options().m_loadPath);

   m_runner->run( m_client, m_sFile, &m_client->options().m_templateWopi );
}


}

/* falhttpd_scripthandler.cpp */
