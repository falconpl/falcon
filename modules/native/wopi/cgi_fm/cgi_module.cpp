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
#include <falcon/process.h>
#include <falcon/vmcontext.h>

namespace Falcon {
namespace WOPI {

ModuleCGI::ModuleCGI():
   ModuleWopi("CGI")
{}

ModuleCGI::~ModuleCGI()
{

}



}
}



/* end of cgi_module.cpp */
