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
#include <falcon/pcode.h>
#include <falcon/symbol.h>
#include <falcon/syntree.h>
#include <falcon/statement.h>
#include <falcon/item.h>
#include <falcon/function.h>
#include <falcon/codeerror.h>

#include <falcon/stream.h>
#include <falcon/stdstreams.h>
#include <falcon/textreader.h>
#include <falcon/textwriter.h>
#include <falcon/transcoder.h>

#include <falcon/genericerror.h>
#include <falcon/locationinfo.h>
#include <falcon/module.h>
#include <falcon/trace.h>
#include <falcon/autocstring.h>
#include <falcon/mt.h>
#include <falcon/globalsymbol.h>

#include <map>
#include <list>

namespace Falcon
{

/** Pair of symbols and the module where they are declared.
 
 Actually, the VM never needs to know where the symbols are decalred.
 This information is used in diagnostics in case of duplicated symbols
 or missing imports.
 
 */
class SymMod {
public:
   const Symbol* m_symbol;
   Module* m_module;

   SymMod( const Symbol* s, Module* m ):
      m_symbol( s ),
      m_module( m )
   {}
};


class ErrorDef
{
public:
   int m_id;
   Module* m_mod;
   const Symbol* m_sym;
   String m_extra;

   ErrorDef( int id, Module* mod, const Symbol* sym, const String& extra ):
      m_id(id),
      m_mod(mod),
      m_sym(sym),
      m_extra(extra)
   {}
};


class VMachine::Private
{
public:
   typedef std::map<String, SymMod> SymbolMap;
   typedef std::list<Module*> ModuleList;
   typedef std::list<ErrorDef> ErrorList;

   ModuleList m_modules;

   // Mutex to lock the error list.
   Mutex m_cserr;
   ErrorList* m_errlist;

   // mutex to lock the exported and imported lists
   Mutex m_csexport;
   SymbolMap m_exported;
   SymbolMap m_imported;

   Private():
      m_errlist( new ErrorList )
   {}

   ~Private()
   {
      delete m_errlist;
   }
};


VMachine::VMachine( Stream* stdIn, Stream* stdOut, Stream* stdErr ):
   m_event(eventNone)
{
   // create the first context
   TRACE( "Virtual machine created at %p", this );
   _p = new Private;
   m_context = new VMContext(this);

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


void VMachine::onError( Error* e )
{
   // for now, just raise.
   throw e;
}


void VMachine::onRaise( const Item& item )
{
   // for now, just wrap and raise.

   //TODO: extract the error if the item is an instance of error.
   Error* e = new GenericError( ErrorParam(e_uncaught,__LINE__)
         .module("VM") );
   e->raised( item );
   throw e;
}


void VMachine::raiseItem( const Item& item )
{
   regA() = item;
   m_event = eventRaise;
}


void VMachine::raiseError( Error* e )
{
   e->scriptize(regA());
   m_event = eventRaise;
}


bool VMachine::run()
{
   MESSAGE( "Run called" );
   m_event = eventNone;
   PARANOID( "Call stack empty", (currentContext()->callDepth() > 0) );

   // for now... then it will be a TLS variable.
   VMContext* ctx = currentContext();
   
   while( true )
   {
      // the code should never be empty.
      // to stop the VM the caller should push a terminator, at least.
      fassert( ! codeEmpty() );

      // BEGIN STEP
      const PStep* ps = ctx->currentCode().m_step;

      try
      {
         ps->apply( ps, ctx );
      }
      catch( Error* e )
      {
         onError( e );
         continue;
      }

      switch( m_event )
      {
         case eventNone: break;

         case eventBreak:
            TRACE( "Hit breakpoint before %s ", location().c_ize() );
            return false;

         case eventComplete:
            MESSAGE( "Run terminated because lower-level complete detected" );
            return true;

         case eventTerminate:
            MESSAGE( "Terminating on explicit termination request" );
            return true;

         case eventReturn:
            MESSAGE( "Retnring on setReturn request" );
            m_event = eventNone;
            return false;

         case eventRaise:
            onRaise( regA() );
            // if we're still alive it means the event was correctly handled
            break;
      }
      // END STEP
   }

   MESSAGE( "Run terminated because of code exaustion" );
   m_event = eventComplete;
   return true;
}


const PStep* VMachine::nextStep() const
{
   MESSAGE( "Next step" );
   if( codeEmpty() )
   {
      return 0;
   }
   PARANOID( "Call stack empty", (currentContext()->callDepth() > 0) );


   CodeFrame& cframe = currentContext()->currentCode();
   const PStep* ps = cframe.m_step;

   if( ps->isComposed() )
   {
      const SynTree* st = static_cast<const SynTree*>(ps);
      return st->at(cframe.m_seqId);
   }
   return ps;
}



String VMachine::report()
{
   register VMContext* ctx = m_context;

   String data = String("Call: ").N( (int32) ctx->callDepth() )
         .A("; Code: ").N((int32)ctx->codeDepth()).A("/").N(ctx->m_topCode->m_seqId)
         .A("; Data: ").N((int32)ctx->dataSize());

   String tmp;

   if( ctx->dataSize() > 0 )
   {
      ctx->topData().describe(tmp);
      data += " (" + tmp + ")";
   }

   data.A("; A: ");
   ctx->m_regA.describe(tmp);
   data += tmp;

   return data;
}


String VMachine::location() const
{
   LocationInfo infos;
   if ( ! location(infos) )
   {
      return "terminated";
   }

   String temp;
   if( infos.m_moduleUri != "" )
   {
      temp = infos.m_moduleUri;
   }
   else
   {
      temp = infos.m_moduleName != "" ? infos.m_moduleName : "<no module>";
   }

   temp += ":" + (infos.m_function == "" ? "<no func>" : infos.m_function);
   if( infos.m_line )
   {
      temp.A(" (").N(infos.m_line);
      if ( infos.m_char )
      {
         temp.A(":").N(infos.m_char);
      }
      temp.A(")");
   }

   return temp;
}


bool VMachine::location( LocationInfo& infos ) const
{
   // location is given by current function and its module plus current source line.
   if( codeEmpty() )
   {
      return false;
   }

   VMContext* vmc = currentContext();

   if( vmc->callDepth() > 0 && vmc->currentFrame().m_function != 0 )
   {
      Function* f = vmc->currentFrame().m_function;
      if ( f->module() != 0 )
      {
         infos.m_moduleName = f->module()->name();
         infos.m_moduleUri = f->module()->uri();
      }
      else
      {
         infos.m_moduleName = "";
         infos.m_moduleUri = "";
      }

      infos.m_function = f->name();
   }
   else 
   {
      infos.m_moduleName = "";
      infos.m_moduleUri = "";
      infos.m_function = "";
   }

   
   const PStep* ps = nextStep();
   if( ps != 0 )
   {
      infos.m_line = ps->line();
      infos.m_char = ps->chr();
   }
   else
   {
      infos.m_line = 0;
      infos.m_char = 0;
   }

   return true;
}


bool VMachine::step()
{
   if ( codeEmpty() )
   {
      MESSAGE( "Step terminated" );
      return true;
   }

   PARANOID( "Call stack empty", (currentContext()->callDepth() > 0) );

   // NOTE: This code must be manually coordinated with vm::run()
   // other solutions, as inline() or macros are either unsafe or
   // clumsy.
   
   // In short, each time vm::run is touched, copy here everything between
   // BEGIN OF STEP - END OF STEP

   // BEGIN OF STEP
   const PStep* ps = currentContext()->currentCode().m_step;
   TRACE( "Step at %s", location().c_ize() );  // this is not in VM::Run
   try
   {
      ps->apply( ps, currentContext() );
   }
   catch( Error* e )
   {
      onError( e );
      return true;
   }

   switch( m_event )
   {
      case eventNone: break;

      case eventBreak:
         TRACE( "Hit breakpoint before line %s ", location().c_ize() );
         return false;

      case eventComplete:
         MESSAGE( "Run terminated because lower-level complete detected" );
         return true;

      case eventTerminate:
         MESSAGE( "Terminating on explicit termination request" );
         return true;

      case eventReturn:
         MESSAGE( "Retnring on setReturn request" );
         m_event = eventNone;
         return false;

      case eventRaise:
         onRaise( regA() );
         // if we're still alive it means the event was correctly handled
         break;
   }
   // END OF STEP

   return codeEmpty();  // more data waiting ?
}


Item* VMachine::findLocalItem( const String& ) const
{
   //TODO
   return 0;
}



void VMachine::link( Module* mod )
{
   // Record the imported symbols.

   // Now save the exported symbols.
   class _rator: public Module::SymbolEnumerator {
   public:
      _rator(VMachine* vm, Module* mod ):
         m_vm(vm),
         m_mod( mod )
      {}

      virtual bool operator()( const Symbol& sym, bool )
      {
         m_vm->addExportedSymbol( m_mod, &sym );
         return true;
      }

   private:
      VMachine* m_vm;
      Module* m_mod;
   };

   _rator rator( this, mod );
   mod->enumerateExports( rator );
}


bool VMachine::addExportedSymbol( Module* mod, const Symbol* sym )
{
   Private::SymbolMap& exp = _p->m_exported;

   _p->m_csexport.lock();
   Private::SymbolMap::iterator iter = exp.find(sym->name());
   // first time around?
   if( iter == exp.end() )
   {
      // insert it
      exp.insert( std::make_pair(sym->name(), SymMod( sym, mod )) );
      //TODO see if there was an imported symbol that is defined here?
      _p->m_csexport.unlock();

      return true;
   }
   else
   {
      // report error
      // what was the place where the symbol was declared?
      SymMod& sm = iter->second;
      _p->m_csexport.unlock();

      String place;
      if( sm.m_module != 0 ) {
         place = sm.m_module->uri() + ".";
      }
      else
      {
         place = "<private>.";
      }

      place += sym->name();

      if( sym->declaredAt() != 0 )
      {
         place.A("(").N(sym->declaredAt()).A(")");
      }

      addLinkError( e_already_def, mod, sym, place );
      return false;
   }
}


void VMachine::addLinkError( int err_id, Module* mod, const Symbol* sym, const String& extra )
{
   _p->m_cserr.lock();
   _p->m_errlist->push_back( ErrorDef( err_id, mod, sym, extra ) );
   _p->m_cserr.unlock();
}


const Symbol* VMachine::findExportedSymbol( const String& name ) const
{
   Private::SymbolMap::iterator iter;

   _p->m_csexport.lock();
   iter = _p->m_exported.find( name );
   if( iter != _p->m_exported.end() )
   {
      const Symbol* sym = iter->second.m_symbol;
      _p->m_csexport.unlock();
      return sym;
   }

   _p->m_csexport.unlock();
   return 0;
}


Error* VMachine::checkRun()
{
   Private::ErrorList* errList;

   // shift the error list thread-safely
   _p->m_cserr.lock();
   // nothing to do?   
   if( _p->m_errlist->size() != 0 )
   {
      _p->m_cserr.unlock();
      return 0;
   }

   // ok we need to
   errList = _p->m_errlist;
   _p->m_errlist = new Private::ErrorList;
   _p->m_cserr.unlock();

   // now we can raise the list.
   Error* err = new GenericError( ErrorParam(e_compile) );  //TODO -- use LinkError
   Private::ErrorList::iterator ei = errList->begin();
   while( ei != errList->end() )
   {
      ErrorDef& def = *ei;
      ErrorParam epar( def.m_id, def.m_sym->declaredAt() );
      if( def.m_mod != 0 )
      {
         epar.module( def.m_mod->uri() );
      }
      else
      {
         epar.module( "<internal>" );
      }
      epar.symbol( def.m_sym->name() );
      epar.extra( def.m_extra );

      err->appendSubError( new CodeError( epar ) );
      ++ei;
   }

   delete errList;

   //TODO -- check imported symbols

   return err;
}

}

/* end of vm.cpp */
