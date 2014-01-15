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
#include <falcon/wopi/scriptrunner.h>
#include <falcon/wopi/errorhandler.h>

namespace Falcon {

class ScriptHandler: public FalhttpdRequestHandler
{
public:
   ScriptHandler( const Falcon::String& sFile, FalhttpdClient* client );
   virtual ~ScriptHandler();
   virtual void serve();

private:
   WOPI::ScriptRunner* m_runner;
};

}

#endif

/* falhttpd_scripthandler.h */
