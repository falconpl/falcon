/*
   FALCON - The Falcon Programming Language.
   FILE: cgi.cpp

   Standalone CGI module for Falcon WOPI.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 28 Feb 2010 17:53:04 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
*/



#include <falcon/module.h>
#include <falcon/wopi/wopi_ext.h>
#include <cgi_request.h>
#include <cgi_reply.h>

#include <cgi_make_streams.h>
#include <falcon/stdstreams.h>
#include <falcon/fstream.h>


#include "cgifm_ext.h"

Falcon::Stream* makeOutputStream()
{
   return new Falcon::StdOutStream;
}


FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

   // initialize the module
   Falcon::Module *self = Falcon::WOPI::wopi_module_init(
         CGIRequest::factory, CGIReply::factory,
         Falcon::CGIRequest_init, Falcon::CGIReply_init
         );

   //Change the name
   self->name( "cgi" );

   return self;
}

/* end of cgi.cpp */
