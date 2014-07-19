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
#include <falcon/stringstream.h>
#include <falcon/modcompiler.h>

#include <set>
#include <map>
#include <list>

#define MAX_TLGEN 2100000000

namespace Falcon {

namespace {
class Breakpoint
{
public:
   int32 m_id;
   int32 m_line;
   String m_modName;
   String m_modPath;
   Module* m_mod;

   bool m_bEnabled;
   bool m_bTemp;

   Breakpoint(int32 id, int32 line, const String& path, const String& name, bool bTemporary, bool bEnabled):
      m_id(id),
      m_line(line),
      m_modName(name),
      m_modPath(path),
      m_mod(0),
      m_bEnabled(bEnabled),
      m_bTemp(bTemporary)
   {}

   Breakpoint(int32 line, bool bTemporary):
      m_id(0),
      m_line(line),
      m_mod(0),
      m_bEnabled(true),
      m_bTemp(bTemporary)
   {}

   Breakpoint(Module* mod, int32 line):
      m_line(line),
      m_mod(mod),
      m_bEnabled(true),
      m_bTemp(true)
   {}

   Breakpoint( const Breakpoint& other ):
      m_id(other.m_id),
      m_line(other.m_line),
      m_modName(other.m_modName),
      m_modPath(other.m_modPath),
      m_mod(other.m_mod),
      m_bEnabled(other.m_bEnabled),
      m_bTemp(other.m_bTemp)
   {}

   ~Breakpoint() {}
};
}

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

   typedef std::multimap<VMContext*, Breakpoint> NegBreakMap;
   NegBreakMap m_negBreaks;
   Mutex m_mtx_negBreaks;

   typedef std::multimap<String, Breakpoint> BreakMap;
   typedef std::map<int, BreakMap::iterator> BreakByIdMap;
   BreakMap m_breaks;
   BreakByIdMap m_breaksById;
   Mutex m_mtx_breaks;

   SynFunc m_loaderFunc;

   Private():
      m_loaderFunc("__loader__")
   {
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

   class PStepRunMain: public PStep
   {
   public:
      PStepRunMain() { apply = apply_; }
      virtual ~PStepRunMain() {}
      virtual void describeTo( String& tgt ) const {
         tgt = "Process::PStepRunMain";
      }

      static void apply_( const PStep*, VMContext* ctx )
      {
         static Log* LOG =  Engine::instance()->log();

         // the module we're working on is at top data stack.
         Module* mod = static_cast<Module*>(ctx->topData().asInst());
         mod->setMain(true);

         Function* mainFunc = mod->getMainFunction();
         if( mainFunc != 0 )
         {
            ctx->popCode();

            LOG->log(Log::fac_engine, Log::lvl_info, String("Launching main script function on: ") + mod->uri() );
            ctx->call( mainFunc );
         }
         else {
            LOG->log(Log::fac_engine, Log::lvl_info, String("Module has no main script function: ") + mod->uri() );
            throw FALCON_SIGN_XERROR(CodeError, e_no_main, .extra(mod->uri()) );
         }
      }
   }
   m_stepRunMain;


   class PStepSetResult: public PStep
      {
      public:
      PStepSetResult() { apply = apply_; }
         virtual ~PStepSetResult() {}
         virtual void describeTo( String& tgt ) const {
            tgt = "Process::PStepSetResult";
         }

         static void apply_( const PStep*, VMContext* ctx )
         {
            static Log* LOG =  Engine::instance()->log();

            LOG->log(Log::fac_engine, Log::lvl_info, "Execution complete" );
            // exit from the top __loader__ frame function
            ctx->returnFrame( ctx->topData() );
         }
      }
      m_stepSetResult;
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
   m_breakCallback(0),
   m_debug(0)
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

   m_context->registerInGC();
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
   m_breakCallback(0),
   m_debug(0)
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

   // we might have already done this, but in that case,
   // there will be no harm.
   // Instead, we absolutely need this if we destroyed the process prior
   // even trying to run it.
   Engine::collector()->unregisterContext(m_context);

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

   mainContext()->reset();
   mainContext()->pushData(Item());
   mainContext()->call(&_p->m_loaderFunc);
   mainContext()->pushCode( &_p->m_stepSetResult );
   mainContext()->pushCode( &_p->m_stepRunMain );

   LOG->log(Log::fac_engine, Log::lvl_info, String("Internally starting loader process on: ") + script.encode() );
   modSpace()->loadModuleInContext( script.encode(), true, true, true, mainContext(),0,true );
   launch();
   return true;
}


/** Starts a String given an input string as a text. */
bool Process::startScript( const String& text, bool isFTD, const String& modName, const String& modPath )
{
   StringStream ss(text);
   TextReader tr (&ss);
   return startScript(&tr, isFTD, modName, modPath );

}

/** Starts a String given a transcoder */
bool Process::startScript( TextReader* tc, bool isFTD, const String& modName, const String& modPath )
{
   static Log* LOG =  Engine::instance()->log();

   if (! checkRunning() ) {
      return false;
   }

   ModCompiler mc;
   Module * mod = mc.compile(tc, modPath, modName, isFTD);
   if( mod == 0 )
   {
      throw mc.makeError();
   }

   mod->setMain(true);

   VMContext* ctx = mainContext();
   ctx->reset();

   Function* mainFunc = mod->getMainFunction();
   if( mainFunc != 0 )
   {
      LOG->log(Log::fac_engine, Log::lvl_info, String("Launching main script function on: ") + modName );
      ctx->call( mainFunc );
   }
   else {
      LOG->log(Log::fac_engine, Log::lvl_info, String("Module has no main script function: ") + modName );
      throw FALCON_SIGN_XERROR(CodeError, e_no_main, .extra(modName) );
   }

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
   ctx->setStatus(VMContext::statusReady);

   // the context might have been registered in GC before, but we try again
   ctx->registerInGC();

   // prepare the GC to handle it. As soon as the GC is done, the context will be handled to the manager.
   vm()->contextManager().onContextReady(ctx);
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

bool Process::inDebug() const
{
   return atomicFetch(m_debug) != 0;
}


void Process::setDebug( bool m )
{
   atomicSet(m_debug, m ? 1 : 0);
}


bool Process::hitBreakpoint( VMContext* ctx, const PStep* ps )
{
   const CallFrame& cf = ctx->currentFrame();

   // no module? -- no breakpoint
   if( cf.m_function == 0 || cf.m_function->module() == 0 )
   {
      return false;
   }

   Module* mod = cf.m_function->module();
   int32 line = ps->line();

   // is there a negative breakpoint?
   {
      _p->m_mtx_negBreaks.lock();
      Private::NegBreakMap::iterator iter = _p->m_negBreaks.find(ctx);
      while( iter != _p->m_negBreaks.end() && iter->first == ctx )
      {
         const Breakpoint& bp = iter->second;

         if( bp.m_mod != mod || (bp.m_line != line && line != 0) )
         {
            // all the negative breaks are temporary, however you can never know in future...
            if( bp.m_bTemp )
            {
               // negative breaks have no id.
               _p->m_negBreaks.erase(iter);
            }
            _p->m_mtx_negBreaks.unlock();

            return true;
         }
         ++iter;
      }
      _p->m_mtx_negBreaks.unlock();
   }

   // Normal breaks
   {
      _p->m_mtx_breaks.lock();
      const String& modName = mod->name();
      Private::BreakMap::iterator iter = _p->m_breaks.find(modName);
      while( iter != _p->m_breaks.end() && iter->first == modName )
      {
         const Breakpoint& bp = iter->second;

         if( bp.m_bEnabled && bp.m_modName == modName && bp.m_line == line )
         {
            if( bp.m_bTemp )
            {
               _p->m_breaksById.erase(bp.m_id); // ok even if not assigned -- will be 0
               _p->m_breaks.erase(iter);
            }
            _p->m_mtx_breaks.unlock();

            return true;
         }

         ++iter;
      }
      _p->m_mtx_breaks.unlock();
   }

   // TODO: Check the breakpoint in forwards.

   return false;
}


void Process::addNegativeBreakpoint( VMContext* ctx, const PStep* ps )
{
   const CallFrame& cf = ctx->currentFrame();

   // no module? -- no breakpoint
   if( cf.m_function == 0 || cf.m_function->module() == 0 )
   {
      return;
   }

   Module* mod = cf.m_function->module();
   int32 line = ps->line();

   _p->m_mtx_negBreaks.lock();
   _p->m_negBreaks.insert(std::make_pair(ctx, Breakpoint(mod, line)) );
   _p->m_mtx_negBreaks.unlock();

}


int Process::addBreakpoint( const String& path, const String& name, int32 line, bool bTemp, bool bEnabled )
{
   int id = 1;

   _p->m_mtx_breaks.lock();
   // get the next useable ID.
   if( ! _p->m_breaksById.empty() )
   {
      id = _p->m_breaksById.rbegin()->first;
      ++id;
   }

   // if the module is pending, the name is empty, so the entry will be inserted at the top of the multi-map
   Private::BreakMap::iterator iter = _p->m_breaks.insert(std::make_pair( name, Breakpoint(id, line, path, name, bTemp, bEnabled ) ) );
   _p->m_breaksById[id] = iter;
   _p->m_mtx_breaks.unlock();

   return id;
}


bool Process::removeBreakpoint( int id )
{
   _p->m_mtx_breaks.lock();
   Private::BreakByIdMap::iterator iid = _p->m_breaksById.find(id);
   if( iid != _p->m_breaksById.end() )
   {
      _p->m_breaks.erase(iid->second);
      _p->m_breaksById.erase(iid);
      _p->m_mtx_breaks.unlock();
      return true;
   }
   _p->m_mtx_breaks.unlock();
   return false;
}


bool Process::enableBreakpoint( int id, bool mode )
{
   _p->m_mtx_breaks.lock();
   Private::BreakByIdMap::iterator iid = _p->m_breaksById.find(id);
   if( iid != _p->m_breaksById.end() )
   {
      Breakpoint& bp = iid->second->second;
      bp.m_bEnabled = mode;
      _p->m_mtx_breaks.unlock();
      return true;
   }
   _p->m_mtx_breaks.unlock();
   return false;
}


void Process::enumerateBreakpoints( BreakpointEnumerator& be )
{
   _p->m_mtx_breaks.lock();
   Private::BreakByIdMap::iterator iid = _p->m_breaksById.begin();
   while( iid != _p->m_breaksById.end() )
   {
      Breakpoint& bp = iid->second->second;
      be(bp.m_id, bp.m_bEnabled, bp.m_modPath, bp.m_modName, bp.m_line, bp.m_bTemp );
      ++iid;
   }
   _p->m_mtx_breaks.unlock();
}


void Process::onModuleAdded( Module* mod )
{
   int32 found = -1;

   // scan unresolved modules.
   _p->m_mtx_breaks.lock();
   if( !_p->m_breaks.empty() )
   {
      Private::BreakMap::iterator iter = _p->m_breaks.begin();

      while( iter != _p->m_breaks.end() && iter->first == "" )
      {
         if( mod->uri() == iter->second.m_modPath )
         {
            // make a copy, we'll need it...
            Breakpoint bp( iter->second );
            found = bp.m_id;
            bp.m_modName = mod->name();
            _p->m_breaks.erase(iter);
            Private::BreakMap::iterator pos =_p->m_breaks.insert( std::make_pair(mod->name(), bp) );
            _p->m_breaksById.erase(found);
            _p->m_breaksById.insert( std::make_pair(found, pos) );
            break;
         }
         ++iter;
      }

   }
   _p->m_mtx_breaks.unlock();

   if( found > 0 )
   {
      Engine::instance()->log()->log( Log::fac_engine, Log::lvl_info, String("Breakpoint ")
               .N(found).A(" for ").A(mod->uri()).A(" resolved with ").A(mod->name()) );
   }
}

}

/* end of process.h */
