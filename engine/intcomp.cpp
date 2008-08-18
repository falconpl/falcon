/*
   FALCON - The Falcon Programming Language.
   FILE: intcomp.cpp

   Complete encapsulation of an incremental interactive compiler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Aug 2008 11:10:24 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/intcomp.h>
#include <falcon/rosstream.h>
#include <falcon/gencode.h>
#include <falcon/runtime.h>
#include <falcon/flcloader.h>
#include <falcon/src_lexer.h>
#include "core_module/core_module.h"

namespace Falcon
{

InteractiveCompiler::InteractiveCompiler( FlcLoader *l, VMachine *vm ):
   Compiler(),
   m_loader( l ),
   m_lmodule( 0 )
{
   if ( vm == 0 )
   {
      m_vm = new VMachine;
      m_vm->link( core_module_init() );
   }
   else
      m_vm = vm;

   m_modVersion = 0x010000;
   m_language = "en_EN";

   m_module = new Module;
   m_module->name( "falcon.intcomp" );
   m_module->engineVersion( FALCON_VERSION_NUM );

   // link this as main and private.
   m_lmodule = m_vm->link( m_module, true, true );

   // as the module is currently empty, we expect the link to work.
   fassert( m_lmodule );

   // set incremental mode for the lexer.
   m_lexer->incremental( true );
   m_tempLine = 1;
}

InteractiveCompiler::~InteractiveCompiler()
{
   delete m_vm;
   m_module->decref();
}

InteractiveCompiler::t_ret_type InteractiveCompiler::compileNext( const String &input )
{
   ROStringStream ss( input );
   return compileNext( &ss );
}

InteractiveCompiler::t_ret_type InteractiveCompiler::compileNext( Stream *input )
{
   // ensure we're talking the same language...
   m_stream = input;
   m_vm->errorHandler( errorHandler() );

   // reset if last code was executed.
   //if ( m_contextSet.size() == 1 )
   {
      m_contextSet.clear();
      m_context.clear();
      m_loops.clear();
      m_functions.clear();
      m_func_ctx.clear();

      delete m_root;
      m_root = new SourceTree;
      pushContextSet( &m_root->statements() );

      // and an empty list for the local function undefined values
      m_statementVals.clear();
      List *l = new List;
      m_statementVals.pushBack( l );
   }

   // reset compilation
   m_enumId = 0;
   m_staticPrefix = 0;
   m_closureContexts = 0;

   // anyhow reset error count
   m_errors = 0;

   // reset the lexer
   m_lexer->reset();
   m_lexer->line( m_tempLine );
   m_lexer->input( m_stream );

   // prepare to unroll changes in the module
   uint32 modSymSize = m_module->symbols().size();

   // parse
   flc_src_parse( this );

   // TODO: move this directly where changed...
   m_module->language( m_language );
   m_module->version( (uint32) m_modVersion );

   // faulty compilation in incremental steps?
   if( m_errors != 0 || m_contextSet.size() != 1 || m_lexer->hasOpenContexts() )
   {
      // unroll changes to the module.
      for (uint32 pos = modSymSize; pos < m_module->symbols().size(); pos ++ )
      {
         Symbol *sym = m_module->symbols().symbolAt( pos );
         m_module->symbolTable().remove( sym->name() );
         delete sym;
      }
      m_module->symbols().resize(modSymSize);
   }

   // some errors during compilation?
   if( m_errors != 0 )
   {
      // do we still need to raise?
      if ( m_delayRaise && m_rootError != 0 && m_errhand != 0 )
      {
         m_errhand->handleError( m_rootError );
         // TODO: Really?
      }

      if ( m_rootError != 0 )
      {
         m_rootError->decref();
         m_rootError = 0;
      }

      // but do not reset errors, the onwer may want to know.
      return e_error;
   }

   if ( m_lexer->hasOpenContexts() )
   {
      return e_incomplete;
   }

   // If the context is not empty, then we have a partial data.
   // more is needed
   if ( m_contextSet.size() != 1 ) {
      return e_more;
   }

   // now we have to decide what to do with the beast.
   // if it's a directive, we already handled it.
   m_tempLine += m_lexer->previousLine();

   // Is it a class?
   if ( ! m_root->classes().empty() )
   {
      StmtClass *cls = static_cast<StmtClass *>( m_root->classes().front() );

      GenCode gencode( module() );
      gencode.generate( m_root );
      // in case of classes, we have to link the constructor,
      // the class itself and eventually the singleton.
      StmtFunction *init = cls->ctorFunction();
      if ( init != 0 )
      {
         if ( ! m_vm->linkCompleteSymbol( init->symbol(), m_lmodule ) )
            return e_vm_error;
      }

      if ( ! m_vm->linkCompleteSymbol( cls->symbol(), m_lmodule ) )
         return e_vm_error;

      Symbol *singleton = cls->singleton();
      if ( singleton != 0 )
      {
         if ( ! m_vm->linkCompleteSymbol( singleton, m_lmodule ) )
            return e_vm_error;
      }

      return e_decl;
   }

   // if it's a function
   if ( ! m_root->functions().empty() )
   {
      StmtFunction *func = static_cast<StmtFunction *>( m_root->functions().front() );

      GenCode gencode( module() );
      gencode.generate( m_root );
      if ( ! m_vm->linkCompleteSymbol( func->symbol(), m_lmodule ) )
      {
         // perform the unroll
         return e_vm_error;
      }

      return e_decl;
   }

   // if it's a statement we must clear the __main__
   // symbol, generate it and re-execute it.
   if( ! m_root->statements().empty() )
   {
      t_ret_type ret;
      Statement *front = m_root->statements().front();

      // was it an expression?
      if( m_root->statements().front()->type() == Statement::t_autoexp )
      {
         // wrap it around a return, so A is not nilled.
         StmtAutoexpr *ae = static_cast<StmtAutoexpr *>( m_root->statements().pop_front() );
         m_root->statements().push_front( new StmtReturn( 1, ae->value()->clone() ) );

         // what kind of expression is it? -- if it's a call, we're interested
         if( ae->value()->asExpr()->type() == Expression::t_funcall )
            ret = e_call;
         else
            ret = e_expression;

         // we don't need the expression anymore.
         delete ae;
      }
      else
         ret = e_statement;

      // generate it
      GenCode gencode( module() );
      gencode.generate( m_root );

      // now, the symbol table must be traversed.
      MapIterator iter = m_module->symbolTable().map().begin();
      bool success = true;
      while( iter.hasCurrent() )
      {
         Symbol *sym = *(Symbol **) iter.currentValue();
         // try to link undefined symbols.

         if ( sym->isUndefined() && m_lmodule->globals().itemAt( sym->itemId() ).isNil() )
         {
            if ( ! m_vm->linkSymbol( sym, m_lmodule ) )
            {
               // but continue to expose other errors as well.
               success = false;
               // prevent searching again
               sym->setGlobal();
            }
         }

         // next symbol
         iter.next();
      }

      // re-link failed?
      if ( ! success )
         return e_error;

      // launch the vm.
      m_vm->launch();
      if ( m_vm->hadError() )
         return e_error;

      return ret;
   }

   return e_nothing;
}

void InteractiveCompiler::addLoad( const String &name, bool isFilename )
{
   m_loader->errorHandler( errorHandler() );

   // try to load the module.
   Module *mod;
   if ( isFilename )
      mod = m_loader->loadFile( name );
   else
      mod = m_loader->loadName( name, m_module->name() );

   if ( mod == 0 )
   {
      // we had an error.
      m_errors++;
      return;
   }

   Compiler::addLoad( name, isFilename );

   // perform a complete linking
   Runtime rt( m_loader, m_vm );
   rt.hasMainModule( false );
   if( ! rt.addModule( mod ) )
   {
      // we had an error
      m_errors++;
   }
   else {
      if ( ! m_vm->link( &rt ) )
      {
         m_errors++;
      }
   }

   mod->decref();
}

}

/* end of intcomp.cpp */
