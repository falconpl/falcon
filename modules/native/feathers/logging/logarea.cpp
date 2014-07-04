/*
   FALCON - The Falcon Programming Language.
   FILE: logarea.cpp

   Logging module -- log area interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Aug 2013 04:43:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "falcon/modules/native/feathers/logging/logarea.cpp"

#include <falcon/mt.h>
#include <set>
#include "logarea.h"
#include "logchannel.h"


namespace Falcon {
namespace Feathers {

class LogArea::Private
{
public:
   mutable Mutex m_mtx_channels;
   typedef std::set<LogChannel*> ChannelSet;
   ChannelSet m_channels;

   Private()
   {}

   ~Private()
   {
      ChannelSet::iterator ic = m_channels.begin();
      while( ic != m_channels.end() )
      {
         LogChannel* chn = *ic;
         chn->decref();
         ic++;
      }
   }

};

//==========================================================
// Log Area
//==========================================================

LogArea::LogArea( const String& name ):
   m_name( name ),
   m_mark(0)
{
   _p = new Private;
}

LogArea::~LogArea()
{
   delete _p;
}


void LogArea::log( uint32 level, const String& source, const String& func, const String& msg, uint32 code ) const
{
   _p->m_mtx_channels.lock();
   Private::ChannelSet::const_iterator ic = _p->m_channels.begin();
   while( ic != _p->m_channels.end() )
   {
      LogChannel* chn = *ic;
      chn->log( this->name(), source, func, level, msg, code );
      ic++;
   }
   _p->m_mtx_channels.unlock();
}

int LogArea::minlog() const
{
   int ml = -1;

   _p->m_mtx_channels.lock();
   Private::ChannelSet::const_iterator ic = _p->m_channels.begin();
   while( ic != _p->m_channels.end() )
   {
      LogChannel* chn = *ic;
      if ( ml < (int) chn->level() )
      {
           ml = chn->level();
      }
      ic++;
   }
   _p->m_mtx_channels.unlock();
   _p->m_mtx_channels.unlock();
   return ml;
}


void LogArea::addChannel( LogChannel* chn )
{
   chn->incref();
   _p->m_mtx_channels.lock();
   if( _p->m_channels.find( chn ) == _p->m_channels.end() )
   {
      _p->m_channels.insert(chn);
      _p->m_mtx_channels.unlock();
   }
   else {
      _p->m_mtx_channels.unlock();
      chn->decref();
   }

}


void LogArea::removeChannel( LogChannel* chn )
{
  _p->m_mtx_channels.lock();
  if( _p->m_channels.erase(chn) > 0 )
  {
     _p->m_mtx_channels.unlock();
     chn->decref();
  }
  else {
     _p->m_mtx_channels.unlock();
  }
}

}
}

/* end of logarea.cpp */
