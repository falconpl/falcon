/*
   FALCON - The Falcon Programming Language.
   FILE: session_manager.cpp

   Falcon Web Oriented Programming Interface

   Session manager.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Feb 2010 23:17:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/session_manager.h>
#include <falcon/sys.h>

#include <deque>
#include <stdlib.h>

namespace Falcon {
namespace WOPI {

//=======================================================================
// Main falcon
//

SessionData::SessionData( const String& SID ):
      m_lastError( 0 ),
      m_sID( SID ),
      m_bInvalid( false )
{
   m_dataLock.item() = SafeItem( new CoreDict( new LinearDict ) );
   m_dataLock.item().asDict()->bless(true);
   m_dataLock.item().asDict()->put( SafeItem( new CoreString( "SID" )),
            SafeItem(new CoreString(sID())) );
}

SessionData::~SessionData()
{
   clearRefs();
}


void SessionData::touch()
{
   m_touchTime = Sys::_seconds();
}


void SessionData::setError( int e )
{
   m_errorDesc = "";
   m_lastError = e;
}

void SessionData::setError( const String& edesc, int e )
{
   m_errorDesc = edesc;
   m_lastError = e;
}


SessionData::WeakRef* SessionData::getWeakRef()
{
   WeakRef* wr = new WeakRef( this );
   m_reflist.push_back( wr );
   return wr;
}

void SessionData::clearRefs()
{
   std::list<WeakRef*>::iterator iter = m_reflist.begin();
   while( iter != m_reflist.end() )  
   {
      (*iter)->onDestroy();
      ++iter;
   }
   m_reflist.clear();
}

//=======================================================================
// Main session manager
//

SessionManager::SessionManager():
      m_nLastToken(0),
      m_nSessionTimeout(0)
{
   srand( time(0) + Sys::_getpid() );
}

SessionManager::~SessionManager()
{
   m_mtx.lock();
   SessionMap::iterator iter = m_smap.begin();
   while( iter != m_smap.end() )
   {
      SessionData* sd = iter->second;
      delete sd;
      ++iter;
   }
   m_mtx.unlock();
}


SessionData* SessionManager::getSession( const Falcon::String& sSID, uint32 token )
{
   SessionData* sd = 0;
   bool bCreated;
   
   // Should we start a new thread for this?
   if( timeout() > 0 )
      expireOldSessions();

   m_mtx.lock();
   SessionMap::iterator iter = m_smap.find( sSID );
   if( iter != m_smap.end() )
   {
      sd = iter->second;
      if ( sd == 0 || sd->isAssigned() )
      {
         m_mtx.unlock();
         return 0;
      }

      sd->assign( token );
      // We must manipulate m_susers in the lock to prevent concurrent update
      // from other threads.
      m_susers[token].push_back( sd->getWeakRef() );

      // now that the session is assigned, we are free to manipulate it outside the lock.
      m_mtx.unlock();

      bCreated = false;
   }
   else
   {
      // create the session (fast)
      sd = createSession( sSID );
      // assign to our maps
      m_smap[sSID] = sd;
      m_susers[token].push_back( sd->getWeakRef() );
      // assign the session
      sd->assign( token );

      // try to resume after unlock
      m_mtx.unlock();

      bCreated = true;
   }

   // can we resume this session?
   if( ! sd->resume() )
   {
      // all useless work.
      m_mtx.lock();
      m_smap.erase( sSID );
      sd->clearRefs();
      m_mtx.unlock();

      //If the session was created, we should have done it.
      if( ! bCreated )
      {
         sd->setInvalid();
      }
      else
      {
         delete sd;
         sd = 0;
      }
   }

   return sd;
}


SessionData* SessionManager::startSession( uint32 token )
{
   SessionData* sd = 0;
   Falcon::String sSID;

   // No one can possibly use this SD as no one can know it.
   sd = createUniqueId( sSID );
   m_mtx.lock();
   m_susers[token].push_back( sd->getWeakRef() );
   // assign the session
   sd->assign( token );
   m_mtx.unlock();

   return sd;
}

SessionData* SessionManager::startSession( uint32 token, const Falcon::String &sSID )
{
	SessionData* sd = 0;

	m_mtx.lock();
	if ( m_smap.find( sSID ) == m_smap.end() )
	{
	   sd = createSession(sSID);
      m_smap[ sSID ] = sd;
      m_susers[token].push_back( sd->getWeakRef() );
      // assign the session
      sd->assign( token );
	}
   m_mtx.unlock();

	// else, it's still 0
   return sd;
}


// release all the sessions associated with this token
bool SessionManager::releaseSessions( uint32 token )
{
   m_mtx.lock();
   // do we have the session?
   SessionUserMap::iterator pos = m_susers.find( token );
   if( pos != m_susers.end() )
   {
      // copy the list of sessions to be closed, so that we can work on it.
      SessionList lCopy = pos->second;
      m_susers.erase( pos );
      m_mtx.unlock();


      SessionList::iterator iter = lCopy.begin();
      while( iter != lCopy.end() )
      {
         SessionData::WeakRef* wsd = *iter;
         SessionData* sd = wsd->get();

         // Still a valid reference?
         if( sd != 0 )
         {
            // store on persistent media
            sd->store();

            // mark as used now
            sd->touch();

            // make available for other requests
            m_mtx.lock();
            if( timeout() > 0 )
            {
               m_expmap.insert( ExpirationMap::value_type( sd->lastTouched() + timeout(), sd->getWeakRef() ) );
            }

            sd->release();
            m_mtx.unlock();
            
         }

         wsd->dropped();
         ++iter;
      }
   }
   else
   {
      m_mtx.unlock();
   }

   return true;
}

bool SessionManager::closeSession( const String& sSID, uint32 token )
{
   m_mtx.lock();
   SessionMap::iterator iter = m_smap.find( sSID );
   if( iter != m_smap.end() )
   {
      SessionData* sd = iter->second;
      m_smap.erase( sSID );
      sd->clearRefs();
      m_mtx.unlock();

      sd->dispose();
      delete sd;
      return true;
   }

   m_mtx.unlock();
   return false;
}


void SessionManager::expireOldSessions()
{
   numeric now = Sys::_seconds();

   std::deque<SessionData*> expiredSessions;
   m_mtx.lock();
   while( ! m_expmap.empty() && m_expmap.begin()->first < now )
   {
      SessionData::WeakRef* wsd = m_expmap.begin()->second;
      SessionData* sd = wsd->get();
      
      // Is the session still alive?
      if ( sd != 0 && sd->lastTouched() + timeout() < now )
      {
         // the data is dead, so we remove it now from the available map
         m_smap.erase( sd->sID() );

         // prevents others (and ourselves) to use it again
         sd->clearRefs();

         // and we push it aside for later clearing
         expiredSessions.push_back( sd );
      }

      // also, take it away from our expired data
      m_expmap.erase( m_expmap.begin() );
   }

   m_mtx.unlock();

   // now we can destroy the expired sessions
   std::deque<SessionData*>::iterator elem = expiredSessions.begin();
   while( elem != expiredSessions.end() )
   {
      SessionData* sd = *elem;
      sd->dispose();
      delete sd;
      ++elem;
   }
}



SessionData* SessionManager::createUniqueId( Falcon::String& sSID )
{
   static const char* alpha="abcdefghjkilmnopqrstuvwxyzABCDEFGHJKILMNOPQRSTUVWXYZ0123456789";
   SessionData* sd = 0;
   
   bool found = false;
   while( ! found )
   {
      //sSID.N( rand() ).N(rand()).N(rand());
      for( int nCount = 0; nCount < 16; nCount++ )
      {
         sSID += alpha[ rand() % 62 ];
      }
      
      m_mtx.lock();
      if ( m_smap.find( sSID ) == m_smap.end() )
      {
         found = true;
         sd = createSession(sSID);
         m_smap[ sSID ] = sd;
      }
      else
      {
         // try again
         sSID.size(0);
      }
      m_mtx.unlock();
   }
   
   return sd;
}


uint32 SessionManager::getSessionToken()
{
   m_mtx.lock();
   uint32 ret = ++m_nLastToken;

   // prevent roll-over error
   if ( ret == 0 )
      ret = ++m_nLastToken;
   m_mtx.unlock();

   return ret;
}


void SessionManager::configFromModule( const Module* mod )
{
   AttribMap* attribs = mod->attributes();
   if( attribs == 0 )
   {
      return;
   }

   VarDef* value = attribs->findAttrib( FALCON_WOPI_SESSION_TO_ATTRIB );
   if( value != 0 && value->isInteger() )
   {
      m_nSessionTimeout = (uint32) value->asInteger();
   }
}


}
}

/* end of session_manager.cpp */
