/*
   FALCON - The Falcon Programming Language.
   FILE: process.h

   Falcon virtual machine -- process entity.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 09 Aug 2012 18:51:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/process.cpp"

#include <falcon/process.h>
#include <falcon/vmcontext.h>
#include <falcon/vm.h>
#include <falcon/mt.h>

#include <falcon/item.h>
#include <falcon/function.h>
#include <falcon/stream.h>
#include <falcon/stdstreams.h>
#include <falcon/textreader.h>
#include <falcon/textwriter.h>
#include <falcon/transcoder.h>
#include <falcon/closure.h>
#include <falcon/modspace.h>
#include <falcon/synfunc.h>
#include <falcon/error.h>
#include <falcon/gclock.h>
#include <falcon/modspace.h>

#include <set>

namespace Falcon {

class Process::Private
{
public:
   typedef std::set<VMContext*> ContextSet;
   ContextSet m_liveContexts;
};



Process::Process( VMachine* owner, ModSpace* ms ):
   m_vm(owner),
   m_context( 0 ),
   m_event( true, false ),
   m_terminated(0),
   m_running(false),
   m_ctxId(0),
   m_error(0),
   m_added(false),
   m_resultLock(0)
{
   _p = new Private;

   // get an ID for this process.
   m_id = m_vm->getNextProcessID();
   m_context = new VMContext(this, 0);
   m_entry = 0;
   if( ms == 0 )
   {
      m_modspace = new ModSpace(this);
   }
   else
   {
      m_modspace = ms;
      ms->incref();
   }

   inheritStreams();
}


Process::Process( VMachine* owner, bool bAdded ):
   m_vm(owner),
   m_context( new VMContext( this ) ),
   m_event( true, false ),
   m_terminated(0),
   m_running(false),
   m_ctxId(0),
   m_error(0),
   m_added(bAdded),
   m_resultLock(0)
{
   _p = new Private;

   // get an ID for this process.
   m_id = m_vm->getNextProcessID();
   m_context = new VMContext(this, 0);
   m_entry = 0;

   m_modspace = new ModSpace(this);
   inheritStreams();
}

void Process::inheritStreams()
{
   // inherit the streams
   m_stdCoder = m_vm->getStdEncoding();

   m_stdIn = m_vm->stdIn();
   m_stdOut = m_vm->stdOut();
   m_stdErr = m_vm->stdErr();
   m_textIn = m_vm->textIn();
   m_textOut = m_vm->textOut();
   m_textErr = m_vm->textErr();

   m_stdIn->incref();
   m_stdOut->incref();
   m_stdErr->incref();
   m_textIn->incref();
   m_textOut->incref();
   m_textErr->incref();
}



Process::~Process() {
   m_context->decref();
   if( m_error != 0 ) {
      m_error->decref();
   }
   if( m_resultLock !=0 ) {
      m_resultLock->dispose();
   }
   delete m_entry;

   delete _p;
}


void Process::terminate()
{
   if( atomicCAS(m_terminated, 0, 1 ) )
   {
      m_mtxContexts.lock();
      Private::ContextSet::iterator iter = _p->m_liveContexts.begin();
      while( iter != _p->m_liveContexts.end() )
      {
         VMContext* ctx = *iter;
         ctx->terminate();
         ++iter;
      }
      m_mtxContexts.unlock();
   }
}


void Process::adoptModSpace( ModSpace* hostSpace )
{
   hostSpace->incref();
   ModSpace* old = m_modspace;
   m_modspace = hostSpace;

   if (old != 0 )
   {
      old->decref();
   }
}

SynFunc* Process::readyEntry()
{
   m_context->reset();
   delete m_entry;
   m_entry = new SynFunc("#Entry");
   m_context->call( m_entry );

   return m_entry;
}


bool Process::start()
{
   if (! checkRunning() ) {
      return false;
   }

   launch();
   return true;
}

bool Process::start( Function* main, int32 pcount, Item const* params )
{
   if (! checkRunning() ) {
      return false;
   }

   //Put a VM termination request here.
   m_context->call(main, pcount, params);
   // launch is to be called after call,
   // as it may stack higher priority calls for base modules.
   launch();
   return true;
}

bool Process::start( Closure* main, int pcount, Item const* params )
{
   if (! checkRunning() ) {
      return false;
   }

   //Put a VM termination request here.
   m_context->call(main, pcount, params );
   // launch is to be called after call,
   // as it may stack higher priority calls for base modules.
   launch();
   return true;
}

bool Process::startItem( Item& main, int pcount, Item const* params )
{
   if (! checkRunning() ) {
      return false;
   }

   // reset the context prior invoking the entry point
   m_context->callItem(main, pcount, params);
   // launch is to be called after call,
   // as it may stack higher priority calls for base modules.
   launch();
   return true;
}


const Item& Process::result() const
{
   return m_result;
}

Item& Process::result()
{
   return m_result;
}


InterruptibleEvent::wait_result_t Process::wait( int32 timeout )
{
   InterruptibleEvent::wait_result_t retval = m_event.wait(timeout);

   if( m_error != 0 ) {
      Error* e = m_error;
      e->incref();
      throw e;
   }

   return retval;
}

void Process::interrupt()
{
   m_event.interrupt();
}

void Process::completed()
{
   m_event.set();
}

void Process::completedWithError( Error* error )
{
   if( m_error !=0 ) {
      m_error->decref();
   }
   m_error = error;
   error->incref();

   m_event.set();
}

void Process::launch()
{
   // The add will eventually launch the process.
   m_vm->addProcess( this, true );
}


void Process::startContext( VMContext* ctx ) {
   ctx->incref();
   m_mtxContexts.lock();
   _p->m_liveContexts.insert( ctx );
   m_mtxContexts.unlock();

   // also, send the context to the manager for immediate execution.
   ctx->incref();
   m_vm->contextManager().readyContexts().add( ctx );
}

void Process::onContextTerminated( VMContext* ctx )
{
   m_mtxContexts.lock();
   Private::ContextSet::iterator iter = _p->m_liveContexts.find( ctx );
   if( iter != _p->m_liveContexts.end() )
   {
      _p->m_liveContexts.erase(iter);
      m_mtxContexts.unlock();
      ctx->decref();
   }
   else {
      m_mtxContexts.unlock();
   }
}

bool Process::checkRunning()
{
   m_mtxRunning.lock();

   if( m_running ) {
      m_mtxRunning.unlock();
      return false;
   }

   m_running = true;
   m_mtxRunning.unlock();

   return true;
}

int32 Process::getNextContextID()
{
   return atomicInc(m_ctxId);
}

void Process::setResult( const Item& value )
{
   // ignore if already terminated.
   if( atomicFetch(m_terminated) != 0 )
   {
      return;
   }

   if( m_resultLock != 0 ) {
      m_resultLock->dispose();
   }

   if ( value.isUser() ) {
      m_resultLock = Engine::collector()->lock(value);
   }
   else {
      m_resultLock = 0;
   }

   m_result = value;
}



void Process::stdIn( Stream* s )
{
   if( s != 0 ) s->incref();
   if( m_stdIn != 0 ) m_stdIn->decref();
   m_stdIn = s;
   m_textIn->changeStream( s );
}


void Process::stdOut( Stream* s )
{
   if( s != 0 ) s->incref();
   if( m_stdOut != 0 ) m_stdOut->decref();
   m_stdOut = s;
   m_textOut->changeStream( s );
}


void Process::stdErr( Stream* s )
{
   if( s != 0 ) s->incref();
   if( m_stdErr != 0 ) m_stdErr->decref();
   m_stdErr = s;
   m_textErr->changeStream( s );
}


bool Process::setStdEncoding( const String& name )
{
   Transcoder* tc = Engine::instance()->getTranscoder(name);
   if( tc == 0 )
   {
      return false;
   }
   m_stdCoder = tc;
   m_bOwnCoder = false;

   m_textIn->setEncoding( tc );
   m_textOut->setEncoding( tc );
   m_textErr->setEncoding( tc );
   return true;
}


void Process::setStdEncoding( Transcoder* ts, bool bOwn )
{
   m_stdCoder = ts;
   m_bOwnCoder = bOwn;

   m_textIn->setEncoding( ts );
   m_textOut->setEncoding( ts );
   m_textErr->setEncoding( ts );
}


}

/* end of process.h */
