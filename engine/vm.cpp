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
#include <falcon/atomic.h>
#include <falcon/modspace.h>
#include <falcon/modloader.h>
#include <falcon/process.h>
#include <falcon/processor.h>

#include <falcon/errors/codeerror.h>
#include <falcon/errors/genericerror.h>

#include <falcon/errors/unserializableerror.h>

#include <map>
#include <list>
#include <vector>

#define INITIAL_GLOBAL_ALLOC 64
#define INCREMENT_GLOBAL_ALLOC 64

#define DEFAULT_CPU_COUNT 4

namespace Falcon
{


class VMachine::Private
{
public:
   typedef std::map<int32, Process*> ProcessMap;

   ProcessMap m_procmap;
   Mutex m_mtxProc;

   typedef std::vector<Processor*> ProcessorVector;

   atomic_int m_proc_id;
   atomic_int m_ctx_id;
   atomic_int m_group_id;

   /**
    * Processors used by this scheduler.
    */
   Mutex m_mtxProcessors;
   ProcessorVector m_processors;

   bool m_terminate;

   Private():
      m_terminate( true )
   {}

   virtual ~Private()
   {
   }

};


VMachine::VMachine( Stream* stdIn, Stream* stdOut, Stream* stdErr )
{
   // create the first context
   TRACE( "Virtual machine created at %p", this );
   _p = new Private;
   m_modspace = new ModSpace(this);

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

   // start the context manager
   m_ctxMan.start();

   // start the processors
   setProcessorCount(0);
   // TODO: start timer and context manager.
}


VMachine::~VMachine()
{
   TRACE( "Virtual machine being destroyed at %p", this );

   // join all the processors
   joinProcessors();

   // stop the context manager
   m_ctxMan.stop();

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
   
   delete m_modspace;

   delete _p;
   
   TRACE( "Virtual machine destroyed at %p", this );
}


void VMachine::joinProcessors()
{
   // ask the processors to terminate
   contextManager().readyContexts().terminateWaiters();

   Private::ProcessorVector::iterator begin, end;
   begin = _p->m_processors.begin();
   end = _p->m_processors.end();
   while( begin != end ) {
      Processor* p = *begin;
      p->join();
      delete p;
      ++begin;
   }

   _p->m_processors.clear();
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


void VMachine::quit()
{
   //TODO
}


Process* VMachine::createProcess()
{
   Process* proc = new Process(this );
   addProcess(proc, false);
   return proc;
}


void VMachine::addProcess( Process* proc, bool launch )
{
   fassert( proc->m_vm == this );

   if( proc->m_vm == this )
   {
      if( ! proc->m_added ) {
         _p->m_procmap.insert( std::make_pair( proc->id(), proc ) );
         proc->incref(); // for the target
         proc->m_added = true;
      }

      if( launch ) {
         // we're assigning the context to the processor/vm/manager system.
         proc->mainContext()->incref();
         // processors are synchronized on the context queue.
         contextManager().readyContexts().add( proc->mainContext() );
      }
   }
}


Process* VMachine::getProcessByID( int32 pid )
{
   _p->m_mtxProc.lock();
   Private::ProcessMap::iterator iter = _p->m_procmap.find(pid);
   if( iter == _p->m_procmap.end() ) {
      _p->m_mtxProc.unlock();
      return iter->second;
   }
   _p->m_mtxProc.unlock();
   return 0;
}


void VMachine::setProcessorCount( int32 count )
{
   //TODO: get the CPU count.
   if( count == 0 ) {
      count = DEFAULT_CPU_COUNT;
   }
   m_processorCount = count;

   updateProcessors();
}


void VMachine::updateProcessors()
{
   // TODO: reduce processors count
   /* Check. Do we have enough processors? */
   _p->m_mtxProcessors.lock();
   while( _p->m_processors.size() < (unsigned) m_processorCount )
   {
      Processor* p = new Processor(_p->m_processors.size(), this);
      _p->m_processors.push_back( p );
      _p->m_mtxProcessors.unlock();

      p->start();

      _p->m_mtxProcessors.lock();
   }
   _p->m_mtxProcessors.unlock();
}

int32 VMachine::getProcessorCount() const
{
   return m_processorCount;
}


int32 VMachine::getNextProcessID()
{
   return atomicInc(_p->m_proc_id);
}

int32 VMachine::getNextContextID()
{
   return atomicInc(_p->m_ctx_id);
}

int32 VMachine::getNextGroupID()
{
   return atomicInc(_p->m_group_id);
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

const Item& VMachine::self() const {
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


Item& VMachine::self() {
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
