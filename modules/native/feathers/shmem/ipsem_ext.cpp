/*
   FALCON - The Falcon Programming Language.
   FILE: ipsem_ext.cpp

   Falcon script interface for Inter-process semaphore.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 11 Nov 2013 16:27:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/native/feathers/shmem/ipsem_ext.cpp"

#include "ipsem_ext.h"
#include "ipsem.h"
#include "shared_ipsem.h"

#include <falcon/function.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/mt.h>
#include <falcon/stdhandlers.h>
#include <falcon/gclock.h>
#include <falcon/classes/classshared.h>

#include <set>
#include <deque>

#define MIN_STACK_SIZE 1024*32
/*#
 @beginmodule shmem
 */

namespace Falcon {
/*#
 @class IPSem
 @brief Inter-process named semaphore
 @param name The name of the semaphore to be opened.
 */

namespace {

/** This class waits for a semaphore to be signaled. */
class SemWaiter: public Runnable
{
public:
   SemWaiter( ClassIPSem* owner );
   ~SemWaiter();

   void wait(SharedIPSem* sem, int64 to);
   void terminate();
   void checkRequest();
   virtual void* run();

private:
   ClassIPSem* m_owner;
   SysThread* m_thread;
   GCLock* m_semLock;

   class Msg
   {
   public:
      SharedIPSem* sem;
      GCLock* lock;
      int64 to;
      Msg(): sem(0),lock(0),to(0) {} // termination message
      Msg(SharedIPSem* s, GCLock* l, int64 t): sem(s),lock(l),to(t) {}
      Msg( const Msg& o): sem(o.sem),lock(o.lock),to(o.to) {}
      ~Msg()  {} // don't release the lock at destructor...
      void detach() {if (lock != 0 ) lock->dispose();}
   };

   typedef std::deque<Msg> MsgList;
   MsgList m_messages;

   Mutex m_mtxMsg;
   Event m_newMsg;

   GCLock* m_classLock;

   void clearMessages();
   void process(SharedIPSem* sem, int64 to);
};


SemWaiter::SemWaiter(ClassIPSem* owner):
         m_owner(owner)
{
   TRACE1("SemWaiter(%p)::SemWaiter(owner:%p)", owner, this);

   // prevent the owner to be destroyed while we're alive.
   m_classLock = Engine::collector()->lock(Item(owner->handler(), owner));

   m_thread = new SysThread(this);
   // start detached: we know when to end.
   m_thread->start(ThreadParams().detached(true).stackSize(MIN_STACK_SIZE));
}

SemWaiter::~SemWaiter()
{
   TRACE1("SemWaiter(%p)::~SemWaiter() destroyed", this);
}

void SemWaiter::wait(SharedIPSem* sem, int64 to)
{
   TRACE1("SemWaiter(%p)::wait(sem:%p, to:%d)", this, sem, (int32) to);

   // we know our owner is the right handler
   GCLock* lock = Engine::collector()->lock(Item(m_owner, sem));

   m_mtxMsg.lock();
   m_messages.push_back(Msg(sem, lock, to));
   m_mtxMsg.unlock();
   m_newMsg.set();
}


void SemWaiter::terminate()
{
   TRACE1("SemWaiter(%p)::terminate()", this);

   m_mtxMsg.lock();
   m_messages.push_back(Msg());
   m_mtxMsg.unlock();
   m_newMsg.set();
}


void* SemWaiter::run()
{
   TRACE1("SemWaiter(%p)::run()", this);

   // Always check if we're wanted.
   while( m_owner->checkNeeded(this) )
   {
      m_newMsg.wait();
      m_mtxMsg.lock();
      if( ! m_messages.empty())
      {
         // copy the incoming message.
         Msg msg( m_messages.front() );
         m_messages.pop_front();
         m_mtxMsg.unlock();

         // then perform the real processing and free the GC lock
         TRACE1("SemWaiter(%p)::run() -- received semaphore %p, %d", this, msg.sem, (int) msg.to);
         if( msg.sem != 0 )
         {
            process(msg.sem, msg.to);
            fassert(msg.lock != 0);
            msg.lock->dispose();
         }
      }
      else {
         // false alarm ...
         m_mtxMsg.unlock();
      }
   }
   TRACE1("SemWaiter(%p)::run() -- terminating", this);

   // we're not going to reference the owner class anymore
   m_classLock->dispose();

   // if we're not needed, this means we're not in the class pool anymore,
   // and this means we won't receive new semaphores -- unlock them.
   clearMessages();

   TRACE1("SemWaiter(%p)::run() -- terminated", this);
   return 0;
}


void SemWaiter::clearMessages()
{
   TRACE1("SemWaiter(%p)::clearMessages()", this);

   MsgList::iterator iter = m_messages.begin();
   MsgList::iterator end = m_messages.begin();
   while( iter != end )
   {
      iter->detach();
      ++iter;
   }
   m_messages.clear();

}

void SemWaiter::process(SharedIPSem* sem, int64 to)
{
   TRACE1("SemWaiter(%p)::process(sem:%p, to:%d)", this, sem, (int32) to);
   fassert( sem != 0 );

   if( sem->semaphore().wait(to) )
   {
      TRACE1("SemWaiter(%p)::process(sem:%p, to:%d) -- signal received", this, sem, (int32) to);
      sem->signal(1);
   }
}

}

//=======================================================================
//
//=======================================================================

class ClassIPSem::Private
{
public:
   Private() {}
   ~Private() {}

   Mutex m_mtxPool;
   typedef std::set<SemWaiter*> WaiterPool;
   WaiterPool m_wpool;

   static const uint32 POOL_SIZE = 4;
};

ClassIPSem::ClassIPSem():
     Class("IPSem")
{
   setParent( Engine::instance()->stdHandlers()->sharedClass() );
   _p = new Private;
}

ClassIPSem::~ClassIPSem()
{
   delete _p;
}

void* ClassIPSem::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

void ClassIPSem::dispose( void* instance ) const
{
   SharedIPSem* mem = static_cast<SharedIPSem*>(instance);
   delete mem;
}

void* ClassIPSem::clone( void* instance ) const
{
   SharedIPSem* sem = static_cast<SharedIPSem*>(instance);
   return new SharedIPSem(*sem);
}

int64 ClassIPSem::occupiedMemory( void* ) const
{
   // account for internal structures.
   return sizeof(SharedIPSem) + 16;
}


void ClassIPSem::gcMarkInstance( void* instance, uint32 mark ) const
{
   SharedIPSem* sem = static_cast<SharedIPSem*>(instance);
   sem->gcMark(mark);
}

bool ClassIPSem::gcCheckInstance( void* instance, uint32 mark ) const
{
   SharedIPSem* sem = static_cast<SharedIPSem*>(instance);
   return sem->currentMark() >= mark;
}

/** Used by an internal class to know if it should stay active or not. */
bool ClassIPSem::checkNeeded( SemWaiter* threadData ) const
{
   bool bNeeded = true;
   _p->m_mtxPool.lock();
   if( _p->m_wpool.size() > Private::POOL_SIZE )
   {
      _p->m_wpool.erase( threadData );
      bNeeded = false;
   }
   _p->m_mtxPool.unlock();

   return bNeeded;
}

}

/* end of ipsem_ext.cpp */
