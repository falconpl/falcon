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
#include <falcon/vm.h>

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

//=======================================================================
// Thread waiting for an IP semaphore to become available.
//=======================================================================

/** This class waits for a semaphore to be signaled. */
class SemWaiter: public Runnable
{
public:
   SemWaiter( const ClassIPSem* owner );
   ~SemWaiter();

   void wait(SharedIPSem* sem, int64 to);
   void terminate();
   void checkRequest();
   virtual void* run();

private:
   const ClassIPSem* m_owner;
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

   void clearMessages();
   void process(SharedIPSem* sem, int64 to);
};


SemWaiter::SemWaiter(const ClassIPSem* owner):
         m_owner(owner)
{
   TRACE1("SemWaiter(%p)::SemWaiter(owner:%p)", owner, this);

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
   bool goOn = true;
   while( goOn )
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
            // as long as we hold the lock on the item,
            // it's handler (our owner) is valid.
            goOn = m_owner->checkNeeded(this);
            // after we use our owner, we can dispose the lock.
            msg.lock->dispose();
         }
         else
         {
            goOn = false;
         }
      }
      else {
         // false alarm ...
         m_mtxMsg.unlock();
      }
   }
   TRACE1("SemWaiter(%p)::run() -- terminating", this);

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

//=======================================================================
// Class IP Semaphore methods
//=======================================================================


static void internal_create( VMContext* ctx, Function* func, int mode )
{
   Item* i_name = ctx->param(0);
   Item* i_public = ctx->param(1);
   if( i_name == 0 || ! i_name->isString() )
   {
      throw func->paramError();
   }

   const String& name = *i_name->asString();
   SharedIPSem* ips = new SharedIPSem(&ctx->vm()->contextManager(), func->methodOf() );
   bool bPublic = i_public == 0? false : i_public->isTrue();

   try {
      switch (mode)
      {
      case 0: ips->semaphore().open(name, bPublic); break;
      case 1: ips->semaphore().openExisting(name); break;
      case 2: ips->semaphore().create(name, bPublic); break;
      }
   }
   catch(...)
   {
      delete ips;
      throw;
   }

   ctx->returnFrame( FALCON_GC_STORE(func->methodOf(), ips) );
}

/*#
 @class IPSem
 @from Shared
 @brief Interprocess semaphore implementation
 @param name The name of the semaphore to be opened
 @optparam public Set to true to create a publicly visible semaphore
 @throw ShmemError on system error or if the semaphore already exists.

 This class implements an inter-process semaphore that
 allows synchronization across different actual O/S processes
 running in the real machine.

 Invoking the constructor is equivalent to invoke the
 @a IPSem.open static method.
 */
FALCON_DECLARE_FUNCTION(init, "name:S, public:[B]")
FALCON_DEFINE_FUNCTION_P1(init)
{
   internal_create( ctx, this, 0 );
}

/*#
 @method open IPSem
 @brief (static) Opens or creates a semaphore.
 @param name The name of the semaphore to be opened.
 @optparam public Set to true to create a publicly visible semaphore
 @throw ShmemError on system error.

  This method opens an interprocess semaphore with the given name,
  eventually creating if it doesn't exists.

  If the @b public parameter is true, the semaphore will be publicly
  visible in a global namespace. On some system, this might require
  administrative privileges.

  @note On MS-Windows, the name of the semaphore may contain a '\\'
  backslash character that will be interpreted as a namespace where
  the semaphore is to be allocated. In this case, the @b public parameter
  will be ignored.
 */
FALCON_DECLARE_FUNCTION(open, "name:S, public:[B]")
FALCON_DEFINE_FUNCTION_P1(open)
{
   internal_create( ctx, this, 0 );
}

/*#
 @method openExisting IPSem
 @brief (static) Opens an existing semaphore.
 @param name The name of the semaphore to be opened.
 @throw ShmemError on system error or if the doesn't exist.

  This method opens an interprocess semaphore with the given name.

  This method differs from @b IPSem.open in the fact that this method will
  raise a ShmemError if the semaphore wasn't already existing (e.g. because
  created by another process).
 */
FALCON_DECLARE_FUNCTION(openExisting, "name:S")
FALCON_DEFINE_FUNCTION_P1(openExisting)
{
   internal_create( ctx, this, 1 );
}


/*#
 @method create IPSem
 @brief (static) Creates a new a semaphore.
 @param name The name of the semaphore to be created.
 @param public True to create a publicly visible semaphore.
 @throw ShmemError on system error or if the semaphore already exists.

  This method tries to create a new semaphore with the given name.
  If a semaphore with the given name already existed, the method
  fails, throwing an error.

  If the parameter @b public is given and true, the semaphore
  is created as publicly visible, otherwise processes created
  by the same user only will be able to open it.
 */
FALCON_DECLARE_FUNCTION(create, "name:S,public:[B]")
FALCON_DEFINE_FUNCTION_P1(create)
{
   internal_create( ctx, this, 2 );
}


/*#
 @method close IPSem
 @brief Closes the given semaphore.
 @optparam remove If true, will remove the semaphore from the system.

 */
FALCON_DECLARE_FUNCTION(close, "remove:[B]")
FALCON_DEFINE_FUNCTION_P1(close)
{
   Item* i_remove = ctx->param(0);
   SharedIPSem* sip = ctx->tself<SharedIPSem>();
   bool bRemove = i_remove == 0 ? false: i_remove->isTrue();
   sip->semaphore().close( bRemove );
   ctx->returnFrame();
}


/*#
 @method post IPSem
 @brief Signals the semaphore.
 @optparam count A positive integer indicating the number of signals to be posted.

   This method signals the semaphore as being waitable, eventually releasing other
   processes using waiters to synchronize on this semaphore.
 */
FALCON_DECLARE_FUNCTION(post, "count:[I]")
FALCON_DEFINE_FUNCTION_P1(post)
{
   Item* i_count = ctx->param(0);

   if( i_count != 0 && ! i_count->isOrdinal() )
   {
      throw paramError();
   }

   int64 count = 1;
   if( i_count != 0 )
   {
      count = i_count->forceInteger();
      if( count <= 0 )
      {
         throw paramError("Invalid range");
      }
   }

   SharedIPSem* sip = ctx->tself<SharedIPSem>();
   while( count > 0 )
   {
      sip->semaphore().post();
      --count;
   }

   ctx->returnFrame();
}

}

//=======================================================================
// Main IP Semaphore class
//=======================================================================

class ClassIPSem::Private
{
public:
   Mutex m_mtxPool;
   typedef std::set<SemWaiter*> WaiterPool;
   WaiterPool m_wpool;

   static const uint32 POOL_SIZE = 4;


   Private() {}
   ~Private() {
      // send a termination request to all the waiters in the pool
      m_mtxPool.lock();
      WaiterPool::iterator iter = m_wpool.begin();
      while( iter != m_wpool.end() )
      {
         (*iter)->terminate();
         ++iter;
      }
      m_mtxPool.unlock();
   }

};

ClassIPSem::ClassIPSem():
     Class("IPSem")
{
   setParent( Engine::instance()->stdHandlers()->sharedClass() );
   _p = new Private;

   setConstuctor( new FALCON_FUNCTION_NAME(init));
   addMethod( new FALCON_FUNCTION_NAME(open), true);
   addMethod( new FALCON_FUNCTION_NAME(create), true);
   addMethod( new FALCON_FUNCTION_NAME(post));
   addMethod( new FALCON_FUNCTION_NAME(close));
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
   // as long as there are instances of us around (i.e. in gc locks),
   // reference ourselves as well, to keep the module alive.
   const_cast<ClassIPSem*>(this)->gcMark(mark);
}

bool ClassIPSem::gcCheckInstance( void* instance, uint32 mark ) const
{
   SharedIPSem* sem = static_cast<SharedIPSem*>(instance);
   return sem->currentMark() >= mark;
}

void ClassIPSem::waitOn( SharedIPSem* ips, int64 to ) const
{
   SemWaiter* waiter = 0;
   _p->m_mtxPool.lock();
   if( ! _p->m_wpool.empty() )
   {
      waiter = *_p->m_wpool.begin();
      _p->m_wpool.erase(_p->m_wpool.begin());
   }
   _p->m_mtxPool.unlock();

   if( waiter == 0 )
   {
      waiter = new SemWaiter(this);
   }

   waiter->wait( ips, to );
}


/** Used by an internal class to know if it should stay active or not. */
bool ClassIPSem::checkNeeded( SemWaiter* threadData ) const
{
   bool bNeeded;
   _p->m_mtxPool.lock();
   if( _p->m_wpool.size() > Private::POOL_SIZE )
   {
      bNeeded = false;
   }
   else
   {
      bNeeded = true;
      _p->m_wpool.insert( threadData );
   }
   _p->m_mtxPool.unlock();

   return bNeeded;
}

}

/* end of ipsem_ext.cpp */
