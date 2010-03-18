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
#include <falcon/modloader.h>
#include <falcon/src_lexer.h>
#include <falcon/transcoding.h>
#include "core_module/core_module.h"

namespace Falcon
{

InteractiveCompiler::InteractiveCompiler( ModuleLoader *l, VMachine *vm ):
   Compiler(),
   m_lmodule( 0 ),
   m_loader( l ),
   m_interactive( true )
{
   m_vm = vm;
   m_vm->incref();

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
   m_tempLine = 1;

   init();
}

InteractiveCompiler::~InteractiveCompiler()
{
   m_module->decref();
   m_vm->decref();
}

InteractiveCompiler::t_ret_type InteractiveCompiler::compileNext( const String &input )
{
   // we split the thing for performance
   // and use stack based object to protect against exception raisal
   if( input.manipulator()->charSize() == 1 )
   {
      ROStringStream ss( input );
      return compileNext( &ss );
   }
   else {
      String temp;
      TranscodeString(input, "utf-16", temp );
      ROStringStream ss( temp );
      TranscoderUTF16 tr( &ss, false );
      return compileNext( &tr );
   }
}

InteractiveCompiler::t_ret_type InteractiveCompiler::compileAll( const String &input )
{
   t_ret_type ret = e_nothing;

   // we split the thing for performance
   // and use stack based object to protect against exception raisal
   if( input.manipulator()->charSize() == 1 )
   {
      ROStringStream ss( input );
      while ( ! ss.eof() )
      {
         // on error, we'll throw.
         ret = compileNext( &ss );
      }
   }
   else {
      String temp;
      TranscodeString(input, "utf-16", temp );
      ROStringStream ss( temp );
      TranscoderUTF16 tr( &ss, false );
      while ( ! tr.eof() )
      {
         // on error, we'll throw.
         ret = compileNext( &tr );
      }
   }

   return ret;
}

InteractiveCompiler::t_ret_type InteractiveCompiler::compileNext( Stream *input )
{
   // ensure we're talking the same language...
   m_stream = input;

   // reset if last code was executed.
   m_contextSet.clear();
   m_context.clear();
   m_loops.clear();
   m_functions.clear();
   m_func_ctx.clear();

   delete m_root;
   m_root = new SourceTree;


   pushContextSet( &m_root->statements() );

   // and an empty list for the local function undefined values
   while( ! m_statementVals.empty() )
   {
      delete (List*) m_statementVals.back();
      m_statementVals.popBack();
   }

   List* l = new List;
   m_statementVals.pushBack( l );

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
   m_lexer->incremental( m_interactive );


   // prepare to unroll changes in the module
   uint32 modSymSize = m_module->symbols().size();

   // parse
   flc_src_parse( this );

   // TODO: move this directly where changed...
   m_module->language( m_language );
   m_module->version( (uint32) m_modVersion );

   // some errors during compilation?
   if( m_errors != 0 )
   {
      // but do not reset errors, the owner may want to know.
      Error *err = m_rootError;
      m_rootError = 0;
      m_module->rollBackSymbols( modSymSize );
      throw err;
   }

   if ( m_lexer->hasOpenContexts() )
   {
      m_module->rollBackSymbols( modSymSize );
      return e_incomplete;
   }

   // If the context is not empty, then we have a partial data.
   // more is needed
   if ( !m_context.empty() || m_contextSet.size() != 1 ) {
      m_module->rollBackSymbols( modSymSize );
      return e_more;
   }

   // now we have to decide what to do with the beast.
   // if it's a directive, we already handled it.
   m_tempLine += m_lexer->previousLine();
   t_ret_type ret = e_nothing;

   // empty?
   if ( m_root->classes().empty() && m_root->functions().empty() && m_root->statements().empty() )
   {
      m_module->rollBackSymbols( modSymSize );
      return e_nothing;
   }

   // if we have some statements, we may need to change the first expression
   // into a return, if we are in interactive mode.
   if( ! m_root->statements().empty() )
   {
      // was it an expression?
      if( m_interactive && m_root->statements().back()->type() == Statement::t_autoexp )
      {
         // wrap it around a return, so A is not nilled.
         StmtAutoexpr *ae = static_cast<StmtAutoexpr *>( m_root->statements().pop_back() );
         m_root->statements().push_back( new StmtReturn( 1, ae->value()->clone() ) );

         // what kind of expression is it? -- if it's a call, we're interested
         if( ae->value()->isExpr() && ae->value()->asExpr()->type() == Expression::t_funcall )
            ret = e_call;
         else
            ret = e_expression;

         // we don't need the expression anymore.
         delete ae;
      }
      else
         ret = e_statement;
   }

   // generate the module
   GenCode gencode( module() );
   gencode.generate( m_root );


   m_lmodule->globals().resize( module()->symbolTable().size() + 1 );

   while ( ! m_root->classes().empty() )
   {
      StmtClass *cls = static_cast<StmtClass *>( m_root->classes().front() );

      // in case of classes, we have to link the constructor,
      // the class itself and eventually the singleton.
      StmtFunction *init = cls->ctorFunction();
      if ( init != 0 )
      {
         ListElement *from_iter = cls->symbol()->getClassDef()->inheritance().begin();
         while( from_iter != 0 )
         {
            const InheritDef *def = (const InheritDef *) from_iter->data();
            const Symbol *parent = def->base();
            // it's just an import
            m_vm->linkSymbol( parent, m_lmodule );
            from_iter = from_iter->next();
         }

         m_vm->linkCompleteSymbol( init->symbol(), m_lmodule );
      }

      m_vm->linkCompleteSymbol( cls->symbol(), m_lmodule );

      Symbol *singleton = cls->singleton();
      if ( singleton != 0 )
      {
         m_vm->linkCompleteSymbol( singleton, m_lmodule );
      }

      if ( ret == e_nothing )
         ret = e_decl;
      delete m_root->classes().pop_front();
   }

   // if it's a function
   while ( ! m_root->functions().empty() )
   {
      StmtFunction *func = static_cast<StmtFunction *>( m_root->functions().front() );
      m_vm->linkCompleteSymbol( func->symbol(), m_lmodule );

      if ( ret == e_nothing )
         ret = e_decl;
      delete m_root->functions().pop_front();
   }

   // now, the symbol table must be traversed.
   MapIterator iter = m_module->symbolTable().map().begin();
   bool success = true;
   while( iter.hasCurrent() )
   {
      Symbol *sym = *(Symbol **) iter.currentValue();
      // try to link undefined symbols.

      if ( sym->isUndefined() )
      {
         try
         {
            m_vm->linkSymbol( sym, m_lmodule );
         }
         catch( Error *e )
         {
            // but continue to expose other errors as well.
            success = false;

            // prevent searching again
            sym->setGlobal();
            raiseError( e );
         }
      }

      // next symbol
      iter.next();
   }

   // re-link failed?
   if ( ! success )
   {
      Error *e = m_rootError;
      m_rootError = 0;
      m_module->rollBackSymbols( modSymSize );
      throw e;
   }

   // finally, link main
   if ( ! m_root->statements().empty() )
   {
      Symbol* msym = module()->findGlobalSymbol("__main__" );
      vm()->linkCompleteSymbol( msym, m_lmodule );
   }

   // launch the vm.
   if ( ret == e_statement || ret == e_call || ret == e_expression )
   {
      try {
         Item* i_main = m_vm->mainModule()->findModuleItem("__main__");
         if( i_main == 0 )
         {
            throw new CodeError( ErrorParam( e_undef_sym, __LINE__ )
                  .origin( e_orig_compiler )
                  .extra( "__main__ (check modules link order/mode)" ) );
         }

         m_vm->reset();
         m_vm->callItem( *i_main, 0 );
      }
      catch( VMEventQuit & )
      {
         ret = e_terminated;
      }
   }

   return ret;
}

void InteractiveCompiler::addNamespace( const String &nspace, const String &alias,
   bool full, bool filename )
{
   loadNow( nspace, filename, true );

   // create also the standard namespace.
   if ( m_errors == 0 )
   {
      Compiler::addNamespace( nspace, alias, full, filename );
   }
}

void InteractiveCompiler::addLoad( const String &name, bool isFilename )
{
   loadNow( name, isFilename, false );

   if ( m_errors == 0 )
   {
      Compiler::addLoad( name, isFilename );
   }
}

void InteractiveCompiler::loadNow( const String &name, bool isFilename, bool bPrivate )
{
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

   // perform a complete linking
   Runtime rt( m_loader, m_vm );
   rt.hasMainModule( false );
   try
   {
      rt.addModule( mod, bPrivate );
      m_vm->link( &rt );
   }
   catch( Error* )
   {
      m_errors++;
      mod->decref();
      throw;
   }
   mod->decref();

}

}

/* end of intcomp.cpp */
