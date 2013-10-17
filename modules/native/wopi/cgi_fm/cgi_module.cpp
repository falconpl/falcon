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
}

ModuleCGI::~ModuleCGI()
{
}


void ModuleCGI::onStartupComplete( VMContext* ctx )
{
   ctx->process()->textOut()->writeLine("Test complete");
   ctx->process()->textOut()->flush();
}


}
}

/* end of cgi_module.cpp */
