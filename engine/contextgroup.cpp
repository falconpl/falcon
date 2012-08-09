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
      if( m_error != 0 ) {
         m_error->decref();
      }
   }
};


ContextGroup::ContextGroup( VMachine* owner, VMContext* parent, uint32 processors ):
   m_owner(owner),
   m_parent( parent ),
   m_processors( processors ),
   m_terminated(0)
{
   // if we have a parent, it must be in the same vm.
   fassert( parent == 0 || parent->vm() == owner );

   m_termEvent = new Shared;
   _p = new Private;
}

ContextGroup::~ContextGroup()
{
   delete _p;
   delete m_termEvent;
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
      ctx->setTerminateEvent();
      // remove sleeping or ready contexts.
      ctx->vm()->removePausedContext(ctx);
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


bool ContextGroup::onContextTerminated( VMContext* )
{
   m_terminated++;
   return m_terminated == (int32) _p->m_contexts.size();
}

void ContextGroup::addContext( VMContext* ctx )
{
   _p->m_contexts.push_back( ctx );
   pushReadyContext(ctx);
}


void ContextGroup::readyAllContexts()
{
   Private::ContextList::const_iterator iter = _p->m_contexts.begin();
   while( iter != _p->m_contexts.end() ) {
      VMContext* ctx = *iter;
      pushReadyContext(ctx);
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


void ContextGroup::pushReadyContext( VMContext* ctx )
{
   _p->m_mtxReadyCtx.lock();
   if( _p->m_running < m_processors )
   {
      _p->m_running++;
      _p->m_mtxReadyCtx.unlock();

      m_owner->pushReadyContext( ctx );
   }
   else {
      _p->m_readyCtx.push_back( ctx );
      _p->m_mtxReadyCtx.unlock();
   }
}

/** Called back when a context is swapped out from a processor. */
void ContextGroup::onContextIdle( VMContext* ctx )
{
   _p->m_mtxReadyCtx.lock();
   --_p->m_running;
   if( ! _p->m_readyCtx.empty() && _p->m_running < m_processors )
   {
     ++_p->m_running;
     ctx = _p->m_readyCtx.front();
     _p->m_readyCtx.pop_front();
     _p->m_mtxReadyCtx.unlock();

     m_owner->pushReadyContext( ctx );
   }
   else {
     _p->m_mtxReadyCtx.unlock();
   }
}

}

/* end of contextgroup.cpp */

