/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_scripthandler.h

   Micro HTTPD server providing Falcon scripts on the web.

   Request handler processing access to directories.
   Mimetype is text/html; charset=utf-8

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Mar 2010 12:18:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALHTTPD_DIRHANDLER_H_
#define FALHTTPD_DIRHANDLER_H_

#include "falhttpd.h"
#include "falhttpd_rh.h"

namespace Falcon {

class DirHandler: public FalhttpdRequestHandler
{
public:
   DirHandler( const Falcon::String& sFile, FalhttpdClient* client );
   virtual ~DirHandler();
   virtual void serve();
};

}

#endif

/* falhttpd_dirhandler.h */
