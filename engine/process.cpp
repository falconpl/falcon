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
#include <falcon/itemdict.h>
#include <falcon/pool.h>
#include <falcon/itemstack.h>
#include <falcon/modloader.h>
#include <falcon/log.h>
#include <falcon/module.h>

#include <set>
#include <map>
#include <list>

#define MAX_TLGEN 2100000000

namespace Falcon {

class Process::Private
{
public:
   typedef std::set<VMContext*> ContextSet;
   ContextSet m_liveContexts;

   typedef std::map <String, String> TransTable;
   TransTable *m_transTable;
   TransTable *m_tempTable;
   Mutex m_mtx_tt;

   typedef std::map <String, GCLock*> ExportMap;
   ExportMap m_exports;
   Mutex m_mtx_exports;

   typedef std::list<GCLock*> CleanupList;
   CleanupList m_cleanups;
   Mutex m_mtx_cleanups;

   Private() {
      m_transTable = 0;
      m_tempTable = 0;
   }

   ~Private() {
      delete m_transTable;
      delete m_tempTable;

      // clear exports
      {
         ExportMap::iterator iter = m_exports.begin();
         while( iter != m_exports.end() )
         {
            GCLock* gl = iter->second;
            gl->dispose();
            ++iter;
         }
      }

      // clear cleanups
      {
         CleanupList::iterator iter = m_cleanups.begin();
         while( iter != m_cleanups.end() )
         {
            GCLock* gl = *iter;
            gl->dispose();
            ++iter;
         }
      }

   }
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
   m_resultLock(0),
   m_mark(0),
   m_tlgen(1),
   m_breakCallback(0)
{
   m_itemPagePool = new Pool;
   m_superglobals = new ItemStack(m_itemPagePool);
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
   m_context( 0 ),
   m_event( true, false ),
   m_terminated(0),
   m_running(false),
   m_ctxId(0),
   m_error(0),
   m_added(bAdded),
   m_resultLock(0),
   m_mark(0),
   m_tlgen(1),
   m_breakCallback(0)
{
   m_itemPagePool = new Pool;
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

   if (m_breakCallback != 0 )
   {
      m_breakCallback->onUnistalled(this);
   }

   m_context->decref();
   m_modspace->decref();
   if( m_error != 0 ) {
      m_error->decref();
   }
   if( m_resultLock !=0 ) {
      m_resultLock->dispose();
   }

   if( m_stdIn != 0 ) m_stdIn->decref();
   if( m_stdOut != 0 ) m_stdOut->decref();
   if( m_stdErr != 0 ) m_stdErr->decref();
   if( m_textIn != 0 ) m_textIn->decref();
   if( m_textOut != 0 ) m_textOut->decref();
   if( m_textErr != 0 ) m_textErr->decref();

   delete m_entry;

   delete _p;
   delete m_itemPagePool;
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

void Process::terminateWithError( Error* error )
{
   if( m_error !=0 ) {
      m_error->decref();
   }
   m_error = error;
   error->incref();

   terminate();
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


void Process::clearError()
{
   if( m_error != 0 )
   {
      m_error->decref();
      m_error = 0;
   }
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


bool Process::startScript( const URI& script, bool addPathToLoadPath )
{
   static Log* LOG =  Engine::instance()->log();

   if (! checkRunning() ) {
      return false;
   }

   if( addPathToLoadPath )
   {
      modSpace()->modLoader()->addDirectoryFront( script.path().fulloc() );
   }

   Process* loadProc = modSpace()->loadModule( script.encode(), true, true, true );

   LOG->log(Log::fac_engine, Log::lvl_info, String("Internally starting loader process on: ") + script.encode() );
   loadProc->start();
   loadProc->wait();
   LOG->log(Log::fac_engine, Log::lvl_info, String("Internally started loader process complete on: ") + script.encode() );

   // get the main module
   Module* mod = static_cast<Module*>(loadProc->result().asInst());
   loadProc->decref();

   Function* mainFunc = mod->getMainFunction();
   if( mainFunc != 0 )
   {
      LOG->log(Log::fac_engine, Log::lvl_info, String("Launching main script function on: ") + script.encode() );

      mainContext()->call( mainFunc );
      launch();
   }
   else {
      LOG->log(Log::fac_engine, Log::lvl_info, String("Module has no main script function: ") + script.encode() );
      throw FALCON_SIGN_XERROR(CodeError, e_no_main, .extra(script.encode()) );
   }

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

void Process::onCompleted()
{
   TRACE( "Process::completed invoked for process %d (%p)", id(), this );
   onCompletedWithError(0);
}

void Process::onCompletedWithError( Error* error )
{
   TRACE( "Process::completedWithError(%p) invoked for process %d (%p)", error, id(), this );

   if( error != 0 )
   {
      if( m_error != 0 ) {
         m_error->decref();
      }

      m_error = error;
      error->incref();
   }

   // is there any termination handler?
   _p->m_mtx_cleanups.lock();
   if( ! _p->m_cleanups.empty() )
   {
      Private::CleanupList copy(_p->m_cleanups);
      _p->m_cleanups.clear();
      _p->m_mtx_cleanups.unlock();

      TRACE( "Process::completedWithError %d (%p) has a cleanup sequence", id(), this );

      // clear the terminated state, in case it was autonomously set
      atomicSet(m_terminated, 0);

      // the context is completed, but still alive.
      m_context->reset();

      Private::CleanupList::iterator iter = copy.begin();
      while( iter != copy.end() )
      {
         GCLock* gl = *iter;
         const Item& clup = gl->item();
         m_context->callItem( clup );
         gl->dispose();
         ++iter;
      }

      // all done during callItem?
      if( m_context->callDepth() > 0 )
      {
         startContext(m_context);
      }
      else {
         atomicSet(m_terminated, 1);
         m_event.set();
      }
   }
   else {
      _p->m_mtx_cleanups.unlock();
      m_event.set();
   }
}

void Process::launch()
{
   // The add will eventually launch the process.
   m_vm->addProcess( this, true );
   startContext( mainContext() );
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
   removeLiveContext( ctx );
}


void Process::removeLiveContext( VMContext* ctx )
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

   m_textIn->setEncoding( tc );
   m_textOut->setEncoding( tc );
   m_textErr->setEncoding( tc );
   return true;
}


void Process::setStdEncoding( Transcoder* ts )
{
   m_stdCoder = ts;

   m_textIn->setEncoding( ts );
   m_textOut->setEncoding( ts );
   m_textErr->setEncoding( ts );
}


bool Process::setTranslationsTable( ItemDict* dict, bool bAdditive )
{
   Private::TransTable* tt = new Private::TransTable;

   try
   {
      class Rator: public ItemDict::Enumerator
      {
      public:
         Rator( Private::TransTable* tt ): m_tt(tt){}
         virtual ~Rator(){}
         virtual void operator()( const Item& key, Item& value )
         {
            if( ! key.isString() || ! value.isString() )
            {
               throw "ops...";
            }
            (*m_tt)[*key.asString()] = *value.asString();
         }
      private:
         Private::TransTable* m_tt;
      };

      Rator rator(tt);
      dict->enumerate(rator);
   }
   catch( ... )
   {
      delete tt;
      return false;
   }

   // if we're here, the stuff is done.
   _p->m_mtx_tt.lock();
   if( bAdditive )
   {
      Private::TransTable::iterator iter = tt->begin();
      while( iter != tt->end() )
      {
         _p->m_transTable->insert( std::make_pair(iter->first, iter->second));
         ++iter;
      }
   }
   else {
      Private::TransTable* tt_old = _p->m_transTable;
      _p->m_transTable = tt;
      tt = tt_old;
   }

   m_tlgen++;
   if( m_tlgen > MAX_TLGEN )
   {
      m_tlgen = 1; // 0 is for newcomers in getTranslation
   }
   _p->m_mtx_tt.unlock();

   // we do this outside the lock.
   delete tt;

   return true;
}

void Process::addTranslation( const String& original, const String& tld )
{
   _p->m_mtx_tt.lock();
   if( _p->m_tempTable == 0 )
   {
      _p->m_tempTable = new Private::TransTable;
   }
   (*_p->m_tempTable)[original] = tld;
   _p->m_mtx_tt.unlock();

}

void Process::commitTranslations( bool bAdditive )
{
   _p->m_mtx_tt.lock();
   Private::TransTable* tt = _p->m_tempTable;
   if( tt == 0 )
   {
      _p->m_mtx_tt.unlock();
      return;
   }

   if( bAdditive )
   {
      Private::TransTable::iterator iter = tt->begin();
      while( iter != tt->end() )
      {
         _p->m_transTable->insert( std::make_pair(iter->first, iter->second));
         ++iter;
      }
   }
   else {
      Private::TransTable* tt_old = _p->m_transTable;
      _p->m_transTable = tt;
      tt = tt_old;
   }

   m_tlgen++;
   if( m_tlgen > MAX_TLGEN )
   {
      m_tlgen = 1; // 0 is for newcomers in getTranslation
   }
   _p->m_tempTable = 0;
   _p->m_mtx_tt.unlock();

   // we do this outside the lock.
   delete tt;
}


void Process::pushCleanup( const Item& code )
{
   GCLock* gl = Engine::collector()->lock(code);
   _p->m_mtx_cleanups.lock();
   _p->m_cleanups.push_back( gl );
   _p->m_mtx_cleanups.unlock();
}


void Process::enumerateTranslations( TranslationEnumerator &te )
{
   _p->m_mtx_tt.lock();
   Private::TransTable* tt = _p->m_transTable;
   if( tt == 0 )
   {
      _p->m_mtx_tt.unlock();
      te.count( 0 );
      return;
   }

   if ( ! te.count( tt->size() ) )
   {
      _p->m_mtx_tt.unlock();
      return;
   }

   Private::TransTable* t1 = new Private::TransTable;
   *t1 = *tt;
   _p->m_mtx_tt.unlock();

   try {
      Private::TransTable::iterator iter = t1->begin();
      while( iter != t1->end() )
      {
         te(iter->first, iter->second);
         ++iter;
      }
      delete t1;
   }
   catch( ... )
   {
      delete t1;
      throw;
   }
}


bool Process::getTranslation( const String& original, String& tld ) const
{
   _p->m_mtx_tt.lock();
   Private::TransTable* tt = _p->m_transTable;
   Private::TransTable::iterator pos = tt->find( original );
   if( pos == tt->end() )
   {
      _p->m_mtx_tt.unlock();
      return false;
   }
   tld = pos->second;
   _p->m_mtx_tt.unlock();

   return true;
}


bool Process::getTranslation( const String& original, String& tld, uint32 &gen) const
{
   _p->m_mtx_tt.lock();
   if( gen == m_tlgen )
   {
      _p->m_mtx_tt.unlock();
      return false;
   }

   gen = m_tlgen;

   Private::TransTable* tt = _p->m_transTable;
   if( tt == 0 )
   {
      _p->m_mtx_tt.unlock();
      tld = original;
      return true;
   }

   Private::TransTable::iterator pos = tt->find( original );
   if( pos != tt->end() )
   {
      tld = pos->second;
      _p->m_mtx_tt.unlock();
   }
   else
   {
      _p->m_mtx_tt.unlock();
      tld = original;
   }

   return true;
}

uint32 Process::getTranslationGeneration() const
{
   _p->m_mtx_tt.lock();
   uint32 gen = m_tlgen;
   _p->m_mtx_tt.unlock();

   return gen;
}


Item* Process::addExport( const String& name, const Item& value )
{
   _p->m_mtx_exports.lock();
   Private::ExportMap::iterator pos = _p->m_exports.find(name);
   if( pos != _p->m_exports.end() )
   {
      _p->m_mtx_exports.unlock();
      return 0;
   }

   Item* ptr = m_superglobals->push();
   _p->m_exports[name] = Engine::collector()->lockPtr(ptr);
   *ptr = value;
   _p->m_mtx_exports.unlock();

   return ptr;
}


bool Process::removeExport( const String& name )
{
   _p->m_mtx_exports.lock();
   Private::ExportMap::size_type count = _p->m_exports.erase(name);
   _p->m_mtx_exports.unlock();

   return count != 0;
}

Item* Process::getExport( const String& name ) const
{
   _p->m_mtx_exports.lock();
   Private::ExportMap::iterator pos = _p->m_exports.find(name);
   if( pos == _p->m_exports.end() )
   {
      _p->m_mtx_exports.unlock();
      return 0;
   }

   Item* target = pos->second->itemPtr();
   _p->m_mtx_exports.unlock();

   return target;
}


Item* Process::updateExport( const String& name, const Item& value, bool &existing ) const
{
   Item* ptr;

   _p->m_mtx_exports.lock();
   Private::ExportMap::iterator pos = _p->m_exports.find(name);
   if( pos != _p->m_exports.end() )
   {
      ptr = pos->second->itemPtr();
      existing = true;
   }
   else {
      ptr = m_superglobals->push();
      _p->m_exports[name] = Engine::collector()->lockPtr(ptr);
      existing = false;
   }
   *ptr = value;
   _p->m_mtx_exports.unlock();

   return ptr;
}

void Process::setBreakCallback( BreakCallback* bcb )
{
   BreakCallback* oldBcb = 0;

   m_mtxBcb.lock();
   oldBcb = m_breakCallback;
   m_breakCallback = bcb;
   m_mtxBcb.unlock();

   if( bcb != 0 )
   {
      bcb->onInstalled(this);
   }

   if (oldBcb != 0 )
   {
      oldBcb->onUnistalled(this);
   }
}


bool Process::onBreakpoint( Processor* prc, VMContext* ctx )
{
   BreakCallback* oldBcb;

   m_mtxBcb.lock();
   oldBcb = m_breakCallback;
   m_mtxBcb.unlock();

   if( oldBcb != 0 )
   {
      oldBcb->onBreak(this, prc, ctx);
      return true;
   }

   return false;
}

}

/* end of process.h */
