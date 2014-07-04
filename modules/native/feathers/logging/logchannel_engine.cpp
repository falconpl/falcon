/*
   FALCON - The Falcon Programming Language.
   FILE: logchannel_tw.cpp

   Logging module -- log channel interface (for textwriter)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Aug 2013 04:43:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "falcon/modules/native/feathers/logging/logchannel_engine.cpp"
#include "logchannel_engine.h"

#include <falcon/engine.h>
#include <falcon/log.h>

namespace Falcon {
namespace Feathers {

LogChannelEngine::LogChannelEngine( int level ):
   LogChannel( level )
{
   Engine::instance()->log()->log( Log::fac_script, Log::lvl_info, "Logging module -- engine log channel starting" );
}

void LogChannelEngine::writeLogEntry( const String& entry, LogChannel::LogMessage* msg )
{
   static Log* log = Engine::instance()->log();
   log->log( Log::fac_script, msg->m_level, entry );
}

LogChannelEngine::~LogChannelEngine()
{
   close();
}

bool LogChannelEngine::close()
{
   if( ! LogChannel::close() )
   {
      return false;
   }

   Engine::instance()->log()->log( Log::fac_script, Log::lvl_info, "Logging module -- engine log channel stopping" );
   return true;
}

}
}

/* end of logchannel_engine.cpp */
