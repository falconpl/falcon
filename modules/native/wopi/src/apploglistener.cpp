/*
   FALCON - The Falcon Programming Language.
   FILE: apploglistener.cpp

   App Log Listener for WOPI.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jan 2014 17:59:47 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/apploglistener.h>
#include <falcon/autocstring.h>
#include <stdio.h>

namespace Falcon {
namespace WOPI {

AppLogListener::AppLogListener()
{
}

AppLogListener::~AppLogListener()
{
}

void AppLogListener::onMessage( int fac, int lvl, const String& message )
{
   String target;
   Log::formatLog(fac,lvl,message,target);
   AutoCString ctarget( target );
   fprintf(stdout, "%s\n", ctarget.c_str() );
}

}
}

/* end of apploglistener.cpp */
