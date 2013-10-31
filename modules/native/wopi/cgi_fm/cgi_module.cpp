/*
   FALCON - The Falcon Programming Language.
   FILE: cgi_module.cpp

   Standalone CGI module for Falcon WOPI
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 14 Oct 2013 00:16:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)
*/

#include "cgi_module.h"
#include <falcon/wopi/replystream.h>
#include <falcon/wopi/request.h>
#include <falcon/wopi/reply.h>
#include <falcon/wopi/stream_ch.h>

#include <falcon/function.h>
#include <falcon/process.h>
#include <falcon/vmcontext.h>
#include <falcon/textwriter.h>

namespace Falcon {
namespace WOPI {

ModuleCGI::ModuleCGI():
   ModuleWopi("CGI")
{
   m_request->parseEnviron();
   m_oldStdOut = 0;
   m_oldStdErr = 0;
}

ModuleCGI::~ModuleCGI()
{
   if( m_oldStdErr != 0 )
   {
      m_process->stdOut(m_oldStdOut);
      m_process->stdErr(m_oldStdErr);
      m_oldStdOut->decref();
      m_oldStdErr->decref();
   }
}


void ModuleCGI::onStartupComplete( VMContext* ctx )
{
   m_process = ctx->process();
   m_oldStdOut = m_process->stdOut();
   m_oldStdErr = m_process->stdErr();

   m_oldStdOut->incref();
   m_oldStdErr->incref();

   m_process->stdOut( new ReplyStream(m_reply, m_oldStdOut, false) );
   m_process->stdErr( new ReplyStream(m_reply, m_oldStdErr, false) );
   // overrides ReplyStream default commit handler
   m_reply->setCommitHandler( new StreamCommitHandler(m_oldStdOut) );

   if( m_request->m_method.compareIgnoreCase("POST") == 0 )
   {
      m_request->parse( ctx->process()->stdIn() );
      m_request->processMultiPartBody();
   }
}


}
}

/* end of cgi_module.cpp */
