/*
   FALCON - The Falcon Programming Language.
   FILE: contextgroup.cpp

   Group of context sharing the same parallel execution.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 05 Aug 2012 22:44:21 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#undef SRC
#define SRC "engine/contextgroup.cpp"

#include <falcon/contextgroup.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/shared.h>
#include <falcon/mt.h>
#include <falcon/itemarray.h>
#include <falcon/fassert.h>
#include <falcon/contextmanager.h>
#include <falcon/error.h>

#include <deque>

namespace Falcon {

class ContextGroup::Private
{
public:
   typedef std::deque<VMContext*> ContextList;

   ContextList m_readyCtx;
   ContextList m_contexts;
   Mutex m_mtxReadyCtx;
   uint32 m_running;

   Error* m_error;

   Private():
      m_running(0),
      m_error(0)
   {}

   ~Private()
   {
      if( m_error != 0 )
      {
         m_error->decref();
      }
   }
};

ContextGroup::ContextGroup():
   m_owner( 0 ),
   m_parent( 0 ),
   m_processors( 0 ),
   m_terminated(0),
   m_refcounter_ContextGroup(1)
{
   _p = new Private;
}


ContextGroup::ContextGroup( VMachine* owner, VMContext* parent, uint32 processors ):
   m_owner(owner),
   m_parent( parent ),
   m_processors( processors ),
   m_terminated(0),
   m_refcounter_ContextGroup(1)
{
   // if we have a parent, it must be in the same vm.
   fassert( parent == 0 || parent->vm() == owner );

   m_termEvent = new Shared( &owner->contextManager() );
   _p = new Private;
}


void ContextGroup::configure( VMachine* owner, VMContext* parent, uint32 processors )
{
   // if we have a parent, it must be in the same vm.
   fassert( parent == 0 || parent->vm() == owner );
   m_owner = owner;
   m_parent = parent;
   m_processors = processors;
   m_termEvent = new Shared( &owner->contextManager() );
}


ContextGroup::~ContextGroup()
{
   delete _p;
   if( m_termEvent != 0 )
   {
      m_termEvent->decref();
   }
}

VMContext* ContextGroup::getContext(uint32 count)
{
   return _p->m_contexts[count];
}

uint32 ContextGroup::getContextCount()
{
   return _p->m_contexts.size();
}


void ContextGroup::setError( Error* error )
{
   // see if we're already throwing an error.
   error->incref();
   _p->m_mtxReadyCtx.lock();
   if( _p->m_error != 0 ) {
      _p->m_mtxReadyCtx.unlock();
      // if so, the first one wins.
      error->decref();
      return;
   }
   _p->m_error = error;
   _p->m_mtxReadyCtx.unlock();

   // signal all the contexts to terminate asap
   // -- the failing context, if any, has already been terminated.
   Private::ContextList::iterator iter = _p->m_contexts.begin();
   while( _p->m_contexts.end() != iter ) {
      VMContext* ctx = *iter;
      // ask the system to kill the contexts.
      ctx->setTerminateEvent();
      ctx->vm()->contextManager().onGroupTerminated( ctx );
      ++iter;
   }
}


Error* ContextGroup::error() const
{
   return _p->m_error;
}


uint32 ContextGroup::runningContexts() const
{
   _p->m_mtxReadyCtx.lock();
   uint32 result = _p->m_running;
   _p->m_mtxReadyCtx.unlock();
   return result;
}

#ifndef NDEBUG
bool ContextGroup::onContextTerminated( VMContext* ctx )
#else
bool ContextGroup::onContextTerminated( VMContext* )
#endif
{
   int terminated = atomicInc(m_terminated);
   int32 size = (int32) _p->m_contexts.size();
   TRACE( "ContextGroup::onContextTerminated on group %p terminated ctx %d, %d/%d",
               this, ctx->id(), terminated, size);
   bool done = terminated >= size;

   if( done ) {
      TRACE( "ContextGroup::onContextTerminated Group %p complete, signaling.", this );
      m_termEvent->signal();
   }

   return done;
}


void ContextGroup::addContext( VMContext* ctx )
{
   _p->m_contexts.push_back( ctx );
}


void ContextGroup::readyAllContexts()
{
   Private::ContextList::const_iterator iter = _p->m_contexts.begin();
   while( iter != _p->m_contexts.end() ) {
      VMContext* ctx = *iter;
      if( onContextReady(ctx) ) {
         ctx->setStatus(VMContext::statusReady);
         ctx->vm()->contextManager().readyContexts().add( ctx );
      }
      ++iter;
   }
}


ItemArray* ContextGroup::results() const
{
   ItemArray* array = new ItemArray( _p->m_contexts.size() );

   Private::ContextList::iterator iter = _p->m_contexts.begin();
   while( _p->m_contexts.end() != iter ) {
      VMContext* ctx = *iter;
      array->append( ctx->topData() );
      ++iter;
   }

   return new ItemArray;
}


VMContext* ContextGroup::onContextIdle()
{
   _p->m_mtxReadyCtx.lock();
   --_p->m_running;
   if( ! _p->m_readyCtx.empty() && _p->m_running < m_processors )
   {
     ++_p->m_running;
     VMContext* ctx = _p->m_readyCtx.front();
     _p->m_readyCtx.pop_front();
     _p->m_mtxReadyCtx.unlock();

     // pass the existing reference to the manager
     return ctx;
   }
   else {
     _p->m_mtxReadyCtx.unlock();
     return 0;
   }
}


bool ContextGroup::onContextReady( VMContext* ctx )
{
   bool result;

   _p->m_mtxReadyCtx.lock();
   if( m_processors > 0  && m_processors <= _p->m_running )
   {
      result = false;
      _p->m_readyCtx.push_front( ctx );
   }
   else {
      ++_p->m_running;
      result = true;
   }
   _p->m_mtxReadyCtx.unlock();

   return result;

}

}

/* end of contextgroup.cpp */

