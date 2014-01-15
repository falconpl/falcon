/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_rh.h

   Micro HTTPD server providing Falcon scripts on the web.

   Abstract interface for request handlers.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 19 Mar 2010 05:13:03 -0700

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALHTTPD_RH_H_
#define FALHTTPD_RH_H_

#include "falhttpd.h"
#include <falcon/wopi/request.h>

namespace Falcon {

class FalhttpdClient;

class FalhttpdRequestHandler
{
public:
   FalhttpdRequestHandler( const Falcon::String& sFile, FalhttpdClient* client );
   virtual ~FalhttpdRequestHandler();
   virtual void serve() = 0;

   const Falcon::String& errorDesc() const { return m_sErrorDesc; }

protected:
   Falcon::String m_sFile;
   Falcon::String m_sErrorDesc;

   FalhttpdClient* m_client;
};

}

#endif

/* falhttpd_rh.h */
