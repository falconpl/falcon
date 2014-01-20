/*
   FALCON - The Falcon Programming Language.
   FILE: log.cpp

   Engine-level pluggable asynchronous logging facility.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Dec 2012 22:29:41 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/log.h>
#include <falcon/syncqueue.h>
#include <falcon/string.h>
#include <falcon/timestamp.h>

#include <set>
#include <deque>
#include <string.h>

namespace Falcon {

class LogMessage {
public:
   union  {
      struct {
         int facility;
         int level;
         String* message;
      }
      data;

      struct {
         Log::Listener* l;
         bool added;
         bool sendMsg;
      }
      listener;
   }
   content;

   bool isData;

   LogMessage( int fac, int lv, String* m )
   {
      isData = true;
      content.data.facility = fac;
      content.data.level = lv;
      content.data.message = m;
   }

   LogMessage( Log::Listener* l, bool added, bool sendMsg )
   {
      isData = false;
      content.listener.l = l;
      content.listener.added = added;
      content.listener.sendMsg = sendMsg;
   }

   LogMessage( const LogMessage& other ) {
      *this = other;
   }

   LogMessage() {}

   LogMessage& operator=(const LogMessage &other) {
      memcpy( &content, &other.content, sizeof(content) );
      isData = other.isData;
      return *this;
   }
};



class Log::Private
{
public:
   Mutex m_mtx;
   typedef std::set<Log::Listener*> ListenerSet;
   typedef std::deque<String*> StringList;

   ListenerSet m_listeners;
   SyncQueue<LogMessage> m_msgList;

   StringList m_stringPool;

   ~Private() {
      {
         ListenerSet::iterator iter = m_listeners.begin();
         ListenerSet::iterator end = m_listeners.end();
         while( iter != end ) {
            (*iter)->decref();
            ++iter;
         }
         m_listeners.clear();
      }

      {
         StringList::iterator iter = m_stringPool.begin();
         StringList::iterator end = m_stringPool.end();
         while( iter != end ) {
            delete *iter;
            ++iter;
         }
         m_stringPool.clear();
      }

   }

};


//======================================================================
// Listener class
//======================================================================

Log::Listener::Listener():
   m_facility(-1),
   m_level(lvl_debug2)
{
}

Log::Listener::~Listener()
{
}



void Log::Listener::logLevel( int lvl )
{
   // we don't care about sync
   m_level = lvl;
}


void Log::Listener::facility( int fac )
{
   m_facility = fac;
}

int Log::Listener::facility() const
{
   return m_facility;
}

int Log::Listener::level() const
{
   return m_level;
}

//======================================================
// Log class
//======================================================

Log::Log()
{
   _p = new Private;
   m_thread = new SysThread( this );
   m_thread->start();

}

Log::~Log()
{
   // send a killer message
   _p->m_msgList.terminateWaiters();
   // wait for completion
   void* result = 0;
   m_thread->join( result );

   // consume last messages.
   LogMessage m;
   while( _p->m_msgList.getST(m) ) {
      if( m.isData ) {
         handleMessage( &m );
      }
   }

   delete _p;
}

void Log::log( int fac, int lvl, const String& message )
{
   String* msg;
   _p->m_mtx.lock();
   if( _p->m_stringPool.empty() ) {
      msg = new String(message);
   }
   else {
      msg = _p->m_stringPool.back();
      // optimal way to copy a string on a preallocated buffer.
      msg->size(0);
      msg->append(message);
      _p->m_stringPool.pop_back();
   }
   _p->m_mtx.unlock();

   _p->m_msgList.add( LogMessage(fac, lvl, msg) );
}


void Log::formatLog( int fac, int lvl, const String& message, String& target )
{
   const char* sFac = facilityToString(fac);
   const char* sLvl = levelToString(lvl);

   TimeStamp ts;
   ts.currentTime();
   ts.toString(target);
   target.A("\t").A(sFac).A(":").A(sLvl).A("\t").A(message);
}

   /**
    * Utility providing a 4 letter level description of a log level.
    *
    * @param lvl The level to be turned into a string.
    * @return The description, or "????" if unknonwn.
    *
    * The possible return values are:
    * - "CRIT"
    * - "ERR "
    * - "WARN"
    * - "INFO"
    * - "DET "
    * - "DBG "
    * - "DBG1"
    * - "DBG2"
    * - "????"
    */
const char* Log::levelToString( int lvl )
{
   switch(lvl)
   {
   case lvl_critical: return "CRIT";
   case lvl_error: return "ERR ";
   case lvl_warn: return "WARN";
   case lvl_info: return "INFO";
   case lvl_detail: return "DET ";
   case lvl_debug: return "DBG ";
   case lvl_debug1: return "DBG1";
   case lvl_debug2: return "DBG2";
   }
   return "????";
}


const char* Log::facilityToString( int fac )
{
   switch(fac)
   {
   case fac_engine: return "E";
   case fac_engine_io: return "I";
   case fac_script: return "S";
   case fac_app: return "A";
   case fac_user: return "U";
   }
   return "?";
}


void Log::addTS( String& target )
{
   TimeStamp ts;
   ts.currentTime();
   ts.toString(target);
}

void Log::addListener( Listener* l, bool msg )
{
   l->incref();
   _p->m_msgList.add( LogMessage(l, true, msg ) );
}


void Log::removeListener( Listener* l, bool msg )
{
   // don't incref; we won't use its pointer till we're sure we have it.
   _p->m_msgList.add( LogMessage(l, false, msg ) );
}

void* Log::run()
{

   int intr = 0;
   LogMessage m;

   while( true )
   {
      _p->m_msgList.get( m, &intr );

      if( intr != 0 ) {
         // we're done.
         break;
      }

      if( m.isData )
      {
         handleMessage(&m);
      }
      else
      {
         handleLogger(&m);
      }
   }

   return 0;
}


void Log::handleMessage( void* data ) {
   LogMessage& m = *(LogMessage*)data;

   Private::ListenerSet::iterator iter = _p->m_listeners.begin();
   Private::ListenerSet::iterator end = _p->m_listeners.end();
   while( iter != end ) {
      Listener* l = *iter;
      if( (l->facility() == fac_all || l->facility() == m.content.data.facility)
         && (l->level() >= m.content.data.level) )
      {
         (*iter)->onMessage( m.content.data.facility, m.content.data.level, *m.content.data.message );
      }
      ++iter;
   }

   _p->m_mtx.lock();
   if( _p->m_stringPool.size() >= STRING_POOL_THRESHOLD ) {
      _p->m_mtx.unlock();
      delete m.content.data.message;
   }
   else {
      _p->m_stringPool.push_back( m.content.data.message );
      _p->m_mtx.unlock();
   }
}


void Log::handleLogger( void* data ) {
   LogMessage& m = *(LogMessage*)data;

   Listener *listener = m.content.listener.l;
   if ( m.content.listener.added ) {
      // new face around?
      if( ! _p->m_listeners.insert(listener).second ) {
         // no? drop the extra ref.
         listener->decref();
      }
      else if (m.content.listener.sendMsg){
         listener->onMessage( fac_engine, lvl_info, "Starting log." );
      }
   }
   else {
      Private::ListenerSet::iterator pos = _p->m_listeners.find( listener );
      if( pos != _p->m_listeners.end() ) {
         _p->m_listeners.erase(pos);
         if (m.content.listener.sendMsg) {
            listener->onMessage( fac_engine, lvl_info, "Stopping log." );
         }
         listener->decref();
      }
   }
}

}

/* end of log.cpp */
