/*
   FALCON - The Falcon Programming Language.
   FILE: cgifm_ext.cpp

   Standalone CGI module for Falcon WOPI - ext declarations
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 28 Feb 2010 17:53:04 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
*/

#include <falcon/vm.h>
#include <falcon/fstream.h>
#include <falcon/module.h>
#include <falcon/sys.h>
#include <falcon/wopi/utils.h>
#include <falcon/wopi/replystream.h>


#include "cgifm_ext.h"

#include <cgi_request.h>
#include <cgi_reply.h>

namespace Falcon
{

void CGIRequest_init( VMachine* vm )
{
   // get ourselves
   CGIRequest* request = dyncast<CGIRequest*>( vm->self().asObject() );

   //vm->stdIn(new Falcon::StdInStream);
   request->PostInitPrepare( vm );
   request->m_cration_time = Falcon::Sys::_seconds();
}


void CGIReply_init( VMachine* vm )
{
   WOPI::Utils::xrandomize();
   
   CGIReply* reply =  Falcon::dyncast<CGIReply*>( vm->self().asObject() );

   // Tell the vm to use a ReplyStream
   vm->stdOut( new WOPI::ReplyStream( reply ) );
   vm->stdErr( new WOPI::ReplyStream( reply ) );

   // And the reply here
   reply->init();
}

}

/* end of cgifm_ext.cpp */
