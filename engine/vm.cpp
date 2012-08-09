/*
   FALCON - The Falcon Programming Language.
   FILE: vm.cpp

   Falcon virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 12 Jan 2011 20:37:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vm.h>
#include <falcon/scheduler.h>
#include <falcon/symbol.h>
#include <falcon/syntree.h>
#include <falcon/statement.h>
#include <falcon/item.h>
#include <falcon/function.h>
#include <falcon/stream.h>
#include <falcon/stdstreams.h>
#include <falcon/textreader.h>
#include <falcon/textwriter.h>
#include <falcon/transcoder.h>
#include <falcon/locationinfo.h>
#include <falcon/module.h>
#include <falcon/trace.h>
#include <falcon/autocstring.h>
#include <falcon/mt.h>
#include <falcon/modspace.h>
#include <falcon/modloader.h>

#include <falcon/errors/codeerror.h>
#include <falcon/errors/genericerror.h>

#include <falcon/errors/unserializableerror.h>

#include <map>
#include <list>
#include <vector>

#define INITIAL_GLOBAL_ALLOC 64
#define INCREMENT_GLOBAL_ALLOC 64

namespace Falcon
{


class VMachine::Private
{
public:

   typedef std::set<ContextGroup*> ContextGroupSet;
   typedef std::deque<VMContext*> ContextList;
   typedef std::vector<Processor*> ProcessorVector;

   Mutex m_mtxGroups;
   Mutex m_mtxReadyContexts;
   Event m_evtCtxReady;

   /**
    * Set of all contexts group.
    */
   ContextGroupSet m_groups;

   /**
    * Context ready to be scheduled.
    */
   ContextList m_readyContexts;

   /**
    * Processors used by this scheduler.
    */
   ProcessorVector m_processors;

   bool m_terminate;

   Private():
      m_evtCtxReady( false, false ),
      m_terminate( true )
   {}

   virtual ~Private()
   {

   }

};


VMachine::VMachine( Stream* stdIn, Stream* stdOut, Stream* stdErr ):
         m_scheduler(0),
         m_lastID(1)
{
   // create the first context
   TRACE( "Virtual machine created at %p", this );
   _p = new Private;
   m_context = new VMContext(this);
   m_modspace = new ModSpace( m_context );

   if ( stdIn == 0 )
   {
      MESSAGE1( "Virtual machine create -- loading duplicate standard input stream" );
      m_stdIn = new StdInStream( true );
   }
   else
   {
      m_stdIn = stdIn;
   }

   if ( stdOut == 0 )
   {
      MESSAGE1( "Virtual machine create -- loading duplicate standard output stream" );
      m_stdOut = new StdOutStream( true );
   }
   else
   {
      m_stdOut = stdOut;
   }

   if ( stdErr == 0 )
   {
      MESSAGE1( "Virtual machine create -- loading duplicate standard error stream" );
      m_stdErr= new StdErrStream( true );
   }
   else
   {
      m_stdErr = stdErr;
   }

   // TODO Determine system transcoder
   m_stdCoder = Engine::instance()->getTranscoder("C");
   fassert( m_stdCoder != 0 );
   m_bOwnCoder = false;

   m_textIn = new TextReader( m_stdIn, m_stdCoder );
   m_textOut = new TextWriter( m_stdOut, m_stdCoder );
   m_textErr = new TextWriter( m_stdErr, m_stdCoder );

#ifdef FALCON_SYSTEM_WIN
   m_textOut->setCRLF(true);
   m_textErr->setCRLF(true);
#endif

   m_textOut->lineFlush(true);
   m_textErr->lineFlush(true);

   m_scheduler = new Scheduler;
}


VMachine::~VMachine()
{
   TRACE( "Virtual machine being destroyed at %p", this );

   delete m_textIn;
   delete m_textOut;
   delete m_textErr;

   delete m_stdIn;
   delete m_stdOut;
   delete m_stdErr;

   if( m_bOwnCoder )
   {
      delete m_stdCoder;
   }
   
   delete m_context;
   delete m_modspace;

   TRACE1( "Deleting scheduler %p", m_scheduler );
   delete m_scheduler;

   delete _p;
   
   TRACE( "Virtual machine destroyed at %p", this );
}


void VMachine::stdIn( Stream* s )
{
   delete m_stdIn;
   m_stdIn = s;
   m_textIn->changeStream( s );
}


void VMachine::stdOut( Stream* s )
{
   delete m_stdOut;
   m_stdOut = s;
   m_textOut->changeStream( s );
}


void VMachine::stdErr( Stream* s )
{
   delete m_stdErr;
   m_stdErr = s;
   m_textErr->changeStream( s );
}


bool VMachine::setStdEncoding( const String& name )
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


void VMachine::setStdEncoding( Transcoder* ts, bool bOwn )
{
   m_stdCoder = ts;
   m_bOwnCoder = bOwn;

   m_textIn->setEncoding( ts );
   m_textOut->setEncoding( ts );
   m_textErr->setEncoding( ts );
}

VMContext* VMachine::getNextReadyContext()
{
   VMContext ctx = 0;

   while(ctx == 0)
   {
      // pick next ready to run context.
      _p->m_mtxReadyContexts.lock();
      // check if we've been asked to stop.
      if( _p->m_terminate )
      {
         _p->m_mtxReadyContexts.unlock();
         break;
      }

      if( _p->m_readyContexts.empty() )
      {
         _p->m_mtxReadyContexts.unlock();
         _p->m_evtCtxReady.wait();
         continue;
      }

      VMContext* ctx = _p->m_readyContexts.front();
      _p->m_readyContexts.pop_front();
      // no more contexts to run?
      if( _p->m_readyContexts.empty() )
      {
         // in case we're asked to terminate, do not switch the signal off.
         if( _p->m_terminate )
         {
            _p->m_mtxReadyContexts.unlock();
            ctx = 0;
            break;
         }

         // else, just tell everyone there's nothing ready.
         _p->m_evtCtxReady.reset();
      }
      _p->m_mtxReadyContexts.unlock();
   }

   return ctx;
}


void VMachine::pushReadyContext(VMContext* ctx)
{
    _p->m_mtxReadyContexts.lock();
    _p->m_readyContexts.push_back( ctx );
    _p->m_evtCtxReady.set();
    _p->m_mtxReadyContexts.unlock();
}


void VMachine::terminate()
{
   _p->m_mtxReadyContexts.lock();
   _p->m_terminate = true;
   _p->m_evtCtxReady.set();
   _p->m_mtxReadyContexts.unlock();
}


void VMachine::addContextGroup(ContextGroup *grp)
{
   // first, save the context.
   _p->m_mtxGroups.lock();
   std::pair<Private::ContextGroupSet::iterator, bool> wasNew =
               _p->m_groups.insert(grp);
   _p->m_mtxGroups.unlock();

   // if it was a new context, ready it for run.
   if( wasNew.second ) {
      grp->readyAllContexts();
   }
}



void VMachine::retval( const Item& v )
{
   // get the thread-specific processor
   Processor* p = Processor::currentProcessor();
   if( p != 0  )
   {
      p->currentContext()->topData() = v;
   }
}


const Item& VMachine::regA() const
{
   static Item fakeA;

   Processor* p = Processor::currentProcessor();
   if( p != 0  )
   {
      return p->currentContext()->topData();
   }
   else {
      return fakeA;
   }
}

Item& VMachine::regA()
{
   static Item fakeA;

   Processor* p = Processor::currentProcessor();
   if( p != 0  )
   {
      return p->currentContext()->topData();
   }
   else {
      return fakeA;
   }
}

const Item& self() const {
   static Item fakeA;

   Processor* p = Processor::currentProcessor();
   if( p != 0  )
   {
      return p->currentContext()->self();
   }
   else {
      return fakeA;
   }
}


Item& self() {
   static Item fakeA;

   Processor* p = Processor::currentProcessor();
   if( p != 0  )
   {
      return p->currentContext()->self();
   }
   else {
      return fakeA;
   }
}

}
/* end of vm.cpp */
