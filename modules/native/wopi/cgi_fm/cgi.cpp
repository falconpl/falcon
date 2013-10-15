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
#include <falcon/stdstreams.h>
#include <falcon/fstream.h>

#include "cgi_module.h"

FALCON_MODULE_DECL
{
   Falcon::WOPI::ModuleCGI* mcgi = new Falcon::WOPI::ModuleCGI;
   return mcgi;
}

/* end of cgi.cpp */
