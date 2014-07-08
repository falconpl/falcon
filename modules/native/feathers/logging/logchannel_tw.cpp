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

#define SRC "falcon/modules/native/feathers/logging/logchannel_tw.cpp"
#include "logchannel_tw.h"

#include <falcon/textwriter.h>

namespace Falcon {
namespace Feathers {

LogChannelTW::LogChannelTW( TextWriter* s, int level ):
   LogChannel( level ),
   m_stream( s ),
   m_bFlushAll( true )
   {
      s->incref();
   }

LogChannelTW::LogChannelTW( TextWriter* s, const String &fmt, int level ):
   LogChannel( fmt, level ),
   m_stream( s ),
   m_bFlushAll( true )
   {
      s->incref();

   }

void LogChannelTW::writeLogEntry( const String& entry, LogChannel::LogMessage* )
{
   m_stream->write(entry);
   m_stream->putChar('\n');

   if( m_bFlushAll )
      m_stream->flush();
}

LogChannelTW::~LogChannelTW()
{
   close();
   m_stream->decref();
}

bool LogChannelTW::close()
{
   if( ! LogChannel::close() )
   {
      return false;
   }

   m_stream->flush();
   m_stream->underlying()->close();
   return true;
}

}
}

/* end of logchannel_tw.cpp */
