/*
   FALCON - The Falcon Programming Language.
   FILE: vmtimer.cpp

   Heart beat timer for processors in the virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 20 Nov 2012 11:41:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vmtimer.h>
#include <falcon/sys.h>
#include <map>
#include <deque>

namespace Falcon {

class VMTimer::Private {
public:
   typedef std::multimap<int64, Token*> TimeMap;
   typedef std::deque<Token*> TokenList;
   TimeMap m_timings;
   TokenList m_expired;

   Private() {}
   ~Private() {}
};


VMTimer::VMTimer()
{
   _p = new Private;
   m_thread = new SysThread(this);
   m_thread->start();
}


VMTimer::~VMTimer()
{
   stop();
}

void VMTimer::stop()
{
   m_mtxStruct.lock();
   if (m_thread == 0)
   {
      m_mtxStruct.unlock();
      return;
   }

   SysThread* th = m_thread;
   m_thread = 0;
   m_mtxStruct.unlock();

   // Signal that the work is over.
   m_newWork.interrupt();

   // wait for the thread to cleanly exit
   void *r = 0;
   th->join(r);
}


VMTimer::Token *VMTimer::setTimeout( uint32 ms, Callback* cb )
{
   int64 now = Sys::_milliseconds();
   return setRandezvous( now + ms, cb );
}


VMTimer::Token *VMTimer::setRandezvous( int64 pointInTime, Callback* cb )
{
   m_mtxTokens.lock();
   Token* t = static_cast<Token*>( m_tokens.get() );

   if( t == 0 ) {
      t = new Token;
      t->assignToPool(&m_tokens);
   }
   m_mtxTokens.unlock();

   t->m_canceled = false;
   t->m_cb = cb;
   t->m_owner = this;

   m_mtxStruct.lock();
   _p->m_timings.insert( std::make_pair( pointInTime, t) );
   m_mtxStruct.unlock();
   m_newWork.set();

   return t;
}

//==========================================================
// Main Thread
//==========================================================

void* VMTimer::run()
{
   int32 rdv = 0;

   while(true)
   {
      if( (m_newWork.wait(rdv) == InterruptibleEvent::wait_interrupted) ) {
         break;
      }

      // check out the expired timings.
      checkExpired(rdv);
   }

   return 0;
}

void VMTimer::checkExpired( int32 &rdv )
{
   // get current system time.
   int64 current = Sys::_milliseconds();
   _p->m_expired.clear();

   // First, get all the expired tokens.
   m_mtxStruct.lock();
   Private::TimeMap::iterator head = _p->m_timings.begin();
   Private::TimeMap::iterator tail = _p->m_timings.end();

   while( head != tail && head->first <= current )
   {
      _p->m_expired.push_back( head->second );
      Private::TimeMap::iterator old = head;
      ++head;
      _p->m_timings.erase(old);
   }

   // update next randez-vous
   if( head != tail )
   {
      rdv = (int)(head->first - current);
      if( rdv < 0 ) {
         // set to a big positive number to prevent endless blocking
         rdv = 0x7FFFFFFF;
      }
   }
   m_mtxStruct.unlock();

   // now the map is free to be updated; we can work on the expired tokens.
   Private::TokenList::iterator ti = _p->m_expired.begin();
   while( ti != _p->m_expired.end() ) {
      Token* current = *ti;

      // tell observer that we're working.
      current->m_notBusy.reset();
      current->m_mtxCanceled.lock();
      if( current->m_canceled ) {
         current->m_mtxCanceled.unlock();
         current->m_notBusy.reset();

         // we're to dispose it
         m_mtxTokens.lock();
         current->dispose();
         m_mtxTokens.unlock();
      }
      else {
         // let the holder call dispose() on cancel()
         current->m_canceled = true;
         current->m_mtxCanceled.unlock();

         // anyhow, the caller of cancel() can't proceed till we're done here.
         (*current->m_cb)();
         current->m_notBusy.set();
      }
      ++ti;
   }

   // we don't need the expired tokens anymore.
   _p->m_expired.clear();
}

//===========================================================
// Token functions
//

VMTimer::Token::Token():
         // not busy event is manual reset and true by default.
         m_notBusy(true,true),
         m_canceled(false)
{
}

VMTimer::Token::~Token()
{
}

void VMTimer::Token::cancel()
{
   // tell the other side that we're out of business
   m_mtxCanceled.lock();
   bool status = m_canceled;
   m_canceled = true;
   m_mtxCanceled.unlock();

   //... and wait in case it was currently working.
   if( status )
   {
      // already processed? -- let's dispose it
      m_owner->m_mtxTokens.lock();
      dispose();
      m_owner->m_mtxTokens.unlock();
   }
   else {
      m_notBusy.wait(-1);
      // the main thread of the timer will dispose it.
   }
}

}

/* end of vmtimer.cpp */

