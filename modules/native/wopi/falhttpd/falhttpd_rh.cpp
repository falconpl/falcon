/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_rh.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   Handler(s) for requests.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 19 Mar 2010 05:13:03 -0700

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include "falhttpd_rh.h"
#include "falhttpd_client.h"

namespace Falcon {

FalhttpdRequestHandler::FalhttpdRequestHandler( const Falcon::String& sFile, FalhttpdClient* cli ):
   m_sFile( sFile ),
   m_client( cli )
{
}


FalhttpdRequestHandler::~FalhttpdRequestHandler()
{
}

}

/* end of falhttpd_rh.cpp */
