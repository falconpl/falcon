/*
   FALCON - The Falcon Programming Language.
   FILE: apache_ll.cpp

   Apache Log Listener for WOPI.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jan 2014 17:59:47 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "apache_ll.h"
#include <falcon/autocstring.h>

using namespace Falcon;

ApacheLogListener::ApacheLogListener( apr_pool_t* pool ):
         m_pool(pool)
{}

ApacheLogListener::~ApacheLogListener()
{
}

void ApacheLogListener::onMessage( int fac, int lvl, const String& message )
{
   int level;
   switch( lvl )
   {
   case Log::lvl_critical: level = APLOG_CRIT; break;
   case Log::lvl_error: level = APLOG_ERR; break;
   case Log::lvl_warn: level = APLOG_WARNING; break;
   case Log::lvl_info: level = APLOG_NOTICE; break;
   case Log::lvl_detail: level = APLOG_INFO; break;
   default: level = APLOG_DEBUG;  break;
   }

   const char* sFac = Log::facilityToString(fac);
   const char* sLvl = Log::levelToString(lvl);
   AutoCString cmsg(message);

   ap_log_perror( APLOG_MARK, level, 0, m_pool,
      "%s:%s %s", sFac, sLvl, cmsg.c_str() );
}


/* end of apache_ll.cpp */
