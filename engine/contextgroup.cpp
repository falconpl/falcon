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

   Error* m_error;

   Private():
      m_error(0)
   {}

   ~Private()
   {
      if( m_error != 0 ) {
         m_error->decref();
      }
   }
};


ContextGroup::ContextGroup( VMachine* owner, VMContext* parent, int32 processors ):
   m_owner(owner),
   m_parent( parent ),
   m_processors( processors ),
   m_running(0),
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
   error->incref();
   _p->m_mtxReadyCtx.lock();
   if( _p->m_error != 0 ) {
      _p->m_mtxReadyCtx.unlock();
      error->decref();
      return;
   }
   _p->m_error = error;
   _p->m_mtxReadyCtx.unlock();

   Private::ContextList::iterator iter = _p->m_contexts.begin();
   while( _p->m_contexts.end() != iter ) {
      VMContext* ctx = *iter;
      ctx->quit();
      ++iter;
   }
}

Error* ContextGroup::error() const
{
   return _p->m_error;
}

bool ContextGroup::terminateContext()
{
   m_terminated++;
   return m_terminated == (int32) _p->m_contexts.size();
}

void ContextGroup::addContext( VMContext* ctx )
{
   _p->m_contexts.push_back( ctx );
   addReadyContext(ctx);
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

VMContext* ContextGroup::nextReadyContext()
{
   VMContext* retval;

   _p->m_mtxReadyCtx.lock();
   if( _p->m_readyCtx.empty() ||
            ( m_processors != 0 && m_running >= m_processors)
            || _p->m_error != 0 )
   {
      retval = 0;
   }
   else {
      retval = _p->m_contexts.front();
      _p->m_contexts.pop_front();
   }
   _p->m_mtxReadyCtx.unlock();

   return retval;
}


void ContextGroup::addReadyContext( VMContext* ctx )
{
   _p->m_mtxReadyCtx.lock();
   _p->m_readyCtx.push_back( ctx );
   _p->m_mtxReadyCtx.unlock();
}

}

/* end of contextgroup.cpp */

