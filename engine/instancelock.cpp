/*
   FALCON - The Falcon Programming Language.
   FILE: instancelock.cpp

   Generic instance-wide lock
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 12 Feb 2013 23:51:42 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/instancelock.h>
#include <falcon/mt.h>

#include <map>

#define INITIAL_LOCK_TOKEN_POOL_COUNT 8
#define MAX_LOCK_TOKEN_POOL_COUNT 32

namespace Falcon {

class InstanceLock::Private {

public:
   Mutex m_mtx;
   typedef std::map<void*, Token*> LockMap;
   LockMap m_locks;

   int32 m_lockCount;
   Token* m_poolHead;

   Private()
   {
      init();
   }

   ~Private() {
      clear();
   }

   void init();
   void clear();

   // to be called with m_mtx held.
   Token* alloc();
   // to be called with m_mtx held.
   void dispose( Token* tk );
};


class InstanceLock::Token
{
public:
   // notice that this is locked inside the Private::m_mtx.
   int m_count;

   Mutex m_tokenLock;
   InstanceLock::Private::LockMap::iterator m_lockPos;
   Token* m_nextLock;
};


void InstanceLock::Private::init()
{
   m_poolHead = 0;

   for( int i = 0; i < INITIAL_LOCK_TOKEN_POOL_COUNT; ++i )
   {
      InstanceLock::Token* tk = new Token;
      tk->m_nextLock = m_poolHead;
      m_poolHead = tk;
   }

   m_lockCount = 0;
}

void InstanceLock::Private::clear()
{
   LockMap::iterator iter = m_locks.begin();
   while (iter != m_locks.end() )
   {
      delete iter->second;
      ++iter;
   }

   InstanceLock::Token* tk = m_poolHead;
   while ( tk != 0 )
   {
      InstanceLock::Token* next = tk->m_nextLock;
      delete tk;
      tk = next;
   }
}

// to be called with m_mtx held.
InstanceLock::Token* InstanceLock::Private::alloc()
{
   InstanceLock::Token* ret;
   if ( m_poolHead != 0 ) {
      ret = m_poolHead;
      m_poolHead = m_poolHead->m_nextLock;
      m_lockCount--;
   }
   else {
      ret = new InstanceLock::Token;
   }

   return ret;
}

// to be called with m_mtx held.
void InstanceLock::Private::dispose( InstanceLock::Token* tk )
{
   if (m_lockCount >= MAX_LOCK_TOKEN_POOL_COUNT )
   {
      delete tk;
   }
   else {
      tk->m_nextLock = m_poolHead;
      m_poolHead = tk;
      m_lockCount++;
   }
}

//=============================================================
// Main class
//

InstanceLock::InstanceLock()
{
   _p = new Private;
}


InstanceLock::~InstanceLock()
{
   delete _p;
}


InstanceLock::Token* InstanceLock::lock( void* instance ) const
{
   Token* tk;
   _p->m_mtx.lock();
   Private::LockMap::iterator pos = _p->m_locks.find(instance);
   if( pos != _p->m_locks.end())
   {
      tk = pos->second;
      tk->m_count++;
   }
   else {
      tk = _p->alloc();
      tk->m_count = 1;
      tk->m_lockPos = _p->m_locks.insert(std::make_pair(instance, tk)).first;
   }
   _p->m_mtx.unlock();

   tk->m_tokenLock.lock();
   return tk;
}


InstanceLock::Token* InstanceLock::trylock( void* instance ) const
{
   Token* tk;
   if (! _p->m_mtx.trylock() )
   {
      return 0;
   }

   Private::LockMap::iterator pos = _p->m_locks.find(instance);
   if( pos != _p->m_locks.end())
   {
      tk = pos->second;

      // it's a try-lock; we can actually cross it with p->mtx
      if( tk->m_tokenLock.trylock() )
      {
         tk->m_count++;
      }
      else
      {
         tk = 0;
      }
   }
   else {
      tk = _p->alloc();
      // we're granted to be the sole owner.
      tk->m_count = 1;
      tk->m_lockPos = _p->m_locks.insert(std::make_pair(instance, tk)).first;

      tk->m_tokenLock.lock();
   }

   _p->m_mtx.unlock();

   return tk;
}


void InstanceLock::unlock( InstanceLock::Token* tk ) const
{
   tk->m_tokenLock.unlock();

   _p->m_mtx.lock();
   if( --tk->m_count == 0 )
   {
      _p->m_locks.erase(tk->m_lockPos);
      _p->dispose(tk);
   }
   _p->m_mtx.unlock();
}


}

/* end of instancelock.cpp */
