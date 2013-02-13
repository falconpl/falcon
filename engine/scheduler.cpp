/*
   FALCON - The Falcon Programming Language.
   FILE: scheduler.cpp

   Timer for delayed activities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 13 Feb 2013 15:17:22 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/scheduler.cpp"

#include <falcon/scheduler.h>
#include <falcon/trace.h>
#include <falcon/fassert.h>
#include <falcon/sys.h>

#include <map>

#define INITIAL_ACTIVITY_POOL_SIZE 8
#define MAXIMUM_ACTIVITY_POOL_SIZE 64


namespace Falcon
{

class Scheduler::Activity {
public:
   Activity() {}
   ~Activity() {}

   Scheduler::t_callback* m_callback;
   void* m_data;
   int64 m_schedule;
   bool m_bCancelable;


   // for pooling.
   Activity* m_next;
};

class Scheduler::Private
{
public:
   // the map is locked by the scheduler MTX
   typedef std::multimap<int64, Activity*> t_activityMap;
   t_activityMap m_activityMap;

   Mutex m_poolMtx;
   Activity* m_poolHead;
   uint32 m_poolSize;

   Private():
      m_poolHead(0),
      m_poolSize(0)
   {
      for( int i = 0; i < INITIAL_ACTIVITY_POOL_SIZE; ++i ) {
         Activity* act = new Activity;
         act->m_next = m_poolHead;
         m_poolHead = act;
      }

      m_poolSize = INITIAL_ACTIVITY_POOL_SIZE;
   }

   ~Private() {
      t_activityMap::iterator iter = m_activityMap.begin();
      while( iter != m_activityMap.end() )
      {
         delete iter->second;
         ++iter;
      }

      Activity* act = m_poolHead;
      while( act != 0 )
      {
         Activity* next = act->m_next;
         delete act;
         act = next;
      }
   }

   Activity* allocate()
   {
      Activity* ret = 0;

      m_poolMtx.lock();
      if( m_poolHead != 0 )
      {
         ret = m_poolHead;
         m_poolHead = ret->m_next;
         m_poolSize--;
      }
      m_poolMtx.unlock();

      if( ret == 0 )
      {
         ret = new Activity;
      }
      return ret;
   }

   void dispose( Activity* a )
   {
      m_poolMtx.lock();
      if( m_poolSize >= MAXIMUM_ACTIVITY_POOL_SIZE )
      {
         m_poolMtx.unlock();
         delete a;
      }
      else
      {
         a->m_next = m_poolHead;
         m_poolHead = a;
         ++m_poolSize;
         m_poolMtx.unlock();
      }

   }
};

//=====================================================
// Main class
//

Scheduler::Scheduler( ):
         m_terminated(false)
{
   _p = new Private;
   m_thread = new SysThread(this);
   m_thread->start();
}

Scheduler::~Scheduler( )
{
   stop();
   delete _p;
}


void Scheduler::stop()
{
   m_mtx.lock();
   if( m_thread != 0 )
   {
      SysThread* thread = m_thread;
      m_thread = 0;
      m_terminated = true;
      m_mtx.unlock();

      m_evtActivity.set();
      void* dummy = 0;
      thread->join(dummy);
   }
   else {
      m_mtx.unlock();
   }
}


Scheduler::Activity* Scheduler::addActivity( uint32 delay, t_callback &cb, void* data, bool cancelable )
{
   return addActivityAt( delay + Sys::_milliseconds(), cb, data, cancelable );
}

Scheduler::Activity* Scheduler::addActivityAt( int64 ts, t_callback &cb, void* data, bool cancelable )
{
   Activity* act = _p->allocate();
   act->m_bCancelable = cancelable;
   act->m_callback = cb;
   act->m_data = data;
   act->m_schedule = ts;

   m_mtx.lock();
   _p->m_activityMap.insert(std::make_pair( act->m_schedule, act ) );
   m_mtx.unlock();

   m_evtActivity.set();

   if( cancelable ) {
      return act;
   }

   return 0;
}


bool Scheduler::cancelActivity( Scheduler::Activity* activity )
{
   m_mtx.lock();
   Private::t_activityMap::iterator iter = _p->m_activityMap.find( activity->m_schedule );
   while( iter != _p->m_activityMap.end() )
   {
      if( iter->second == activity )
      {
         _p->m_activityMap.erase(iter);
         m_mtx.unlock();

         _p->dispose(activity);
         return true;
      }
      ++iter;
   }
   m_mtx.unlock();

   // already gone.
   _p->dispose(activity);
   return false;
}

void Scheduler::completeActivity( Scheduler::Activity* activity )
{
   // we presume the activity is not in the scheduled map anymore,
   // as this method should be called by the callback only,
   // and the callback is called after having descheduled the activity

   // (however, do a minimal check to avoid silly crashes)
   if( activity->m_bCancelable ) {
      _p->dispose(activity);
   }
}


void* Scheduler::run()
{
   int64 currentTime = Sys::_milliseconds();
   int64 nextSchedule = 0;
   Activity* current = 0;

   while( true )
   {
      // wait for something to do.
      m_mtx.lock();
      while(_p->m_activityMap.empty() && ! m_terminated )
      {
         m_mtx.unlock();

         m_evtActivity.wait();
         currentTime = Sys::_milliseconds();

         m_mtx.lock();
      }

      // are we done?
      if( m_terminated )
      {
         m_mtx.unlock();
         break;
      }

      // still locked
      current = _p->m_activityMap.begin()->second;
      if( current->m_schedule <= currentTime )
      {
         // handle this
         _p->m_activityMap.erase( _p->m_activityMap.begin() );
         if( _p->m_activityMap.empty() ) {
            nextSchedule = -1;
         }
         else {
            nextSchedule = _p->m_activityMap.begin()->first;
         }
      }
      else {
         nextSchedule = current->m_schedule;
         current = 0;
      }

      m_mtx.unlock();

      // Here perform the callback.
      if ( current != 0 )
      {
         current->m_callback( current->m_data, current );

         // If the caller didn't request a copy of it, we must dispose it
         if( ! current->m_bCancelable )
         {
            _p->dispose( current );
         }
         // else, it's responsibility of the caller to clean it
      }

      // shall we wait?
      if( nextSchedule > 0 )
      {
         // we don't need to wait here for nextSchedule == -1
         // as it's done when the map is empty, and that will lead us
         // to wait forever on the topmost loop
         m_evtActivity.wait( nextSchedule - currentTime );
         currentTime = Sys::_milliseconds();
      }

   }

   return 0;
}

}


/* scheduler.cpp */
