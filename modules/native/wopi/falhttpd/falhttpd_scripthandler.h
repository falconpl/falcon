/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_scripthandler.h

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


#ifndef FALHTTPD_SCRIPTHANDLER_H_
#define FALHTTPD_SCRIPTHANDLER_H_

#include "falhttpd.h"
#include "falhttpd_rh.h"

class ScriptHandler: public FalhttpdRequestHandler
{
public:
   ScriptHandler( const Falcon::String& sFile, FalhttpdClient* client );
   virtual ~ScriptHandler();
   virtual void serve( Falcon::WOPI::Request* req );

};

#endif

/* falhttpd_scripthandler.h */
