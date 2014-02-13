/*
   FALCON - The Falcon Programming Language.
   FILE: intcompiler.cpp

   Interactive compiler - step by step dynamic compiler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Apr 2011 21:57:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/intcompiler.cpp"

#include <falcon/intcompiler.h>
#include <falcon/module.h>
#include <falcon/symbol.h>
#include <falcon/textwriter.h>

#include <falcon/stringstream.h>
#include <falcon/textreader.h>
#include <falcon/textwriter.h>
#include <falcon/statement.h>

#include <falcon/parser/parser.h>
#include <falcon/sp/sourcelexer.h>
#include <falcon/error.h>
#include <falcon/modspace.h>

#include <falcon/expression.h>
#include <falcon/falconclass.h>
#include <falcon/hyperclass.h>
#include <falcon/modloader.h>
#include <falcon/stdsteps.h>
#include <falcon/stderrors.h>
#include <falcon/vm.h>

#include <falcon/psteps/exprvalue.h>

#include <falcon/datareader.h>
#include <falcon/importdef.h>
#include <falcon/synclasses_id.h>
#include <falcon/process.h>

#include <falcon/attribute_helper.h>


namespace Falcon {

//=======================================================================
// context callback
//

IntCompiler::Context::Context(IntCompiler* owner):
   ModCompiler::Context( owner )
{
}

IntCompiler::Context::~Context()
{
   // nothing to do
}


void IntCompiler::Context::onNewStatement( TreeStep* )
{
   // override the base class and do nothing
}


void IntCompiler::Context::onCloseFunc( Function* function )
{
   ModCompiler::Context::onCloseFunc( function );
   static_cast<IntCompiler*>(m_owner)->m_currentMantra = function;
}


void IntCompiler::Context::onCloseClass( Class* cls, bool isObj )
{
   static StdSteps* steps = Engine::instance()->stdSteps();
   ModCompiler::Context::onCloseClass(cls, isObj );

   Mantra* current;
   FalconClass* fcls = static_cast<FalconClass*>(cls);
   current = fcls;
   VMContext* vmctx = static_cast<IntCompiler*>(m_owner)->m_vmctx;

   try
   {
      if( ! fcls->construct(vmctx) )
      {
         current = fcls->hyperConstruct();
      }
      static_cast<IntCompiler*>(m_owner)->m_currentMantra = current;

      // initialize the object now.
      if ( isObj )
      {
         vmctx->pushCode( &steps->m_fillInstance );
         vmctx->callItem( Item(current->handler(), current) );
      }

   }
   catch( Error* e )
   {
      m_owner->sp().addError( e );
      e->decref();
   }
}


bool IntCompiler::Context::onAttribute(const String& name, TreeStep* generator, Mantra* target )
{
   SourceParser& sp = m_owner->sp();
   if( target == 0 )
   {
      sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
   }
   else
   {
      VMContext* vmctx = static_cast<IntCompiler*>(m_owner)->m_vmctx;
      return attribute_helper( vmctx, name, generator, target );

   }

   return true;
}


void IntCompiler::Context::onLoad( const String& path, bool isFsPath )
{
   SourceParser& sp = m_owner->sp();
   if( ! static_cast<IntCompiler*>(m_owner)->m_bAllowDirective )
   {
      sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      return;
   }

   // get the module space
   ModSpace* theSpace = m_owner->module()->modSpace();

   // do we have a module?
   Error* err = 0;
   bool status = m_owner->module()->addLoad( path, isFsPath, err, sp.currentLexer()->line() );
   if( ! status )
   {
      sp.addError( err );
      return;
   }

   // just adding it top the module won't be of any help.
   VMContext* vmctx = static_cast<IntCompiler*>(m_owner)->m_vmctx;
   theSpace->resolveDeps( vmctx, m_owner->module() );
}


bool IntCompiler::Context::onImportFrom( ImportDef* def )
{
   SourceParser& sp = m_owner->sp();
   if( ! static_cast<IntCompiler*>(m_owner)->m_bAllowDirective )
   {
      sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      return false;
   }

   // have we already a module group for the module?
   // get the module space
   VMContext* cctx = static_cast<IntCompiler*>(m_owner)->m_vmctx;
   Module* mod = m_owner->module();
   ModSpace* ms = cctx->process()->modSpace();
   Error* err = 0;
   mod->addImport( def, err );
   if( err != 0 ) {
      sp.addError(err);
   }
   else {
      ms->resolveDeps( cctx, mod );
   }
   return true;
}


void IntCompiler::Context::onExport(const String& symName )
{
   SourceParser& sp = m_owner->sp();
   if( ! static_cast<IntCompiler*>(m_owner)->m_bAllowDirective )
   {
      sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      return;
   }

   Module* mod = m_owner->module();

   const Symbol* sym = Engine::getSymbol(symName);
   // do we have a module?
   bool status = mod->globals().exportGlobal( sym ) != 0;

   // already exported?
   if( ! status )
   {
      sp.addError( e_export_already, m_owner->sp().currentSource(),
           sp.currentLexer()->line(), 0, 0 );
      sym->decref();
      return;
   }

   status = mod->modSpace()->exportSymbol( sym, mod->globals().getValue(sym) );

   if( ! status )
   {
      sp.addError( e_export_already, sp.currentSource(),
                   sp.currentLexer()->line(), 0, 0 );
   }

   sym->decref();
}


void IntCompiler::Context::onDirective(const String&, const String&)
{
   // TODO
   SourceParser& sp = m_owner->sp();
   if( ! static_cast<IntCompiler*>(m_owner)->m_bAllowDirective )
   {
      sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      return;
   }
}


void IntCompiler::Context::onGlobal( const String& name )
{
   SourceParser& sp = m_owner->sp();
   if( ! static_cast<IntCompiler*>(m_owner)->m_bAllowDirective )
   {
      sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      return;
   }

   ModCompiler::Context::onGlobal( name );
}

bool IntCompiler::Context::onGlobalAccessed( const String& name )
{
   bool isLocal = ModCompiler::Context::onGlobalAccessed(name);

   // if the variable is extern, resolve it immediately
   if( ! isLocal )
   {
      Module* mod = m_owner->module();
      Item* value = mod->resolveGlobally(name);

      if (value == 0)
      {
         SourceParser& sp = m_owner->sp();
         sp.addError(
                  new LinkError(
                           ErrorParam(e_undef_sym, __LINE__, SRC )
                           .extra(name)
                           .module(sp.currentSource())
                           .line( sp.currentLine() )
                           .origin(ErrorParam::e_orig_compiler) ) );
      }
   }

   return isLocal;
}


void IntCompiler::Context::onInputOver()
{
   // we usually do nothing here.
}

//=======================================================================
// Main class
//

IntCompiler::IntCompiler( bool allowDirective ):
   ModCompiler( new IntCompiler::Context( this ) ),
   m_currentTree(0),
   m_bAllowDirective( allowDirective )
{
   // Prepare the compiler and the context.
   m_sp.setContext(m_ctx);
   m_sp.interactive(true);
   m_compf = 0;

   m_lexer = new SourceLexer( "(interactive)", &m_sp );
   m_sp.pushLexer( m_lexer );
}


IntCompiler::~IntCompiler()
{
   delete m_currentTree;
}


void IntCompiler::setCompilationContext( Function* function, Module* mod, VMContext* vmctx )
{
   m_compf = function;
   m_module = mod;
   m_vmctx = vmctx;
}


IntCompiler::t_compile_status IntCompiler::compileNext( TextReader* input, SynTree*& code, Mantra*& definition )
{
   MESSAGE( "IntCompiler::compileNext" );

   m_lexer->setReader(input);
   t_compile_status status;

   // create a new syntree if it was emptied
   if( m_currentTree == 0 )
   {
      m_currentTree = new SynTree;
      m_ctx->openMain( m_currentTree );
      m_sp.pushState( "Main", false );
   }

   // if there is a compilation error, throw it
   m_sp.step();

   if( m_sp.hasErrors() ) {
      throw m_sp.makeError();
   }

   if( isComplete() )
   {
      if( ! m_currentTree->empty() )
      {
         if ( m_currentTree->at(0)->category() == TreeStep::e_cat_expression )
         {
            // this requires evaluation; but is this a direct call?
            Expression* expr = static_cast<Expression*>(m_currentTree->at(0));
            if( expr != 0 && expr->handler()->userFlags() == FALCON_SYNCLASS_ID_CALLFUNC )
            {
               status = e_expr_call;
            }
            else
            {
               status = e_expression;
            }
         }
         else {
            status = e_statement;
         }
         // else ret can stay ok

         m_sp.reset();
         // where to put the tree now?
         // it might be reflected?
         code = m_currentTree;
         m_currentTree = 0;
      }
      else {
         status= m_currentMantra == 0 ? e_incomplete : e_definition;
         definition = m_currentMantra;
      }
   }
   else {
      status = e_incomplete;
   }

   return status;
}


void IntCompiler::resetTree()
{
   m_sp.reset();
   delete m_currentTree;
   delete m_currentMantra;
   m_currentTree = 0;
   m_currentMantra = 0;
}


bool IntCompiler::isComplete() const
{
   return m_sp.isComplete() && m_ctx->isCompleteStatement();
}

}

/* end of intcompiler.h */
