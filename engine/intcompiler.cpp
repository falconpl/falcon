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
#include <falcon/psteps/stmtautoexpr.h>

#include <falcon/parser/parser.h>
#include <falcon/sp/sourcelexer.h>
#include <falcon/error.h>
#include <falcon/modspace.h>

#include <falcon/expression.h>
#include <falcon/falconclass.h>
#include <falcon/hyperclass.h>
#include <falcon/modloader.h>
#include <falcon/requirement.h>

#include <falcon/errors/genericerror.h>
#include <falcon/errors/syntaxerror.h>
#include <falcon/errors/codeerror.h>
#include <falcon/errors/ioerror.h>

#include <falcon/vm.h>

#include <falcon/psteps/exprvalue.h>

#include <falcon/datareader.h>
#include <falcon/importdef.h>
#include <falcon/synclasses_id.h>
#include <falcon/process.h>

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


void IntCompiler::Context::onCloseFunc( Function* function )
{
   ModCompiler::Context::onCloseFunc( function );
   static_cast<IntCompiler*>(m_owner)->m_currentMantra = function;
}


void IntCompiler::Context::onCloseClass( Class* cls, bool isObj )
{
   ModCompiler::Context::onCloseClass(cls, isObj );

   Mantra* current;
   FalconClass* fcls = static_cast<FalconClass*>(cls);
   current = fcls;
   if( fcls->missingParents() == 0 )
   {
      try
      {

         if( ! fcls->construct() )
         {
            current = fcls->hyperConstruct();
         }
         static_cast<IntCompiler*>(m_owner)->m_currentMantra = current;
      }
      catch( Error* e )
      {
         m_owner->sp().addError( e );
         e->decref();
      }
   }

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
   if( m_owner->module()->addLoad( path, isFsPath ) )
   {
      sp.addError( e_load_already, sp.currentSource(), sp.currentLine()-1, 0, 0, path );
      return;
   }

   // just adding it top the module won't be of any help.
   theSpace->loadModule( path, isFsPath, static_cast<IntCompiler*>(m_owner)->m_vmctx );
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
   ModSpace* ms = cctx->vm()->modSpace();
   mod->addImport( def );
   ms->resolveDeps( cctx, mod );

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

   // do we have a module?
   Variable* sym = 0;
   bool already;
   sym = mod->addExport( symName, already );

   // already exported?
   if( already )
   {
      sp.addError( e_export_already, m_owner->sp().currentSource(),
           sym->declaredAt(), 0, 0 );
      return;
   }

   Error* e = mod->modSpace()->exportSymbol( mod, symName, *sym );
   if( e != 0 )
   {
      sp.addError( e );
      e->decref();
   }
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


void IntCompiler::Context::onGlobal( const String& )
{
   // TODO
   SourceParser& sp = m_owner->sp();
   if( ! static_cast<IntCompiler*>(m_owner)->m_bAllowDirective )
   {
      sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      return;
   }
}


void IntCompiler::Context::onRequirement( Requirement* req )
{
   // the incremental compiler cannot store requirements.
   delete req;
   m_owner->sp().addError( e_undef_sym, m_owner->sp().currentSource(),
                req->sourceRef().line(), req->sourceRef().chr(), 0, req->name() );
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
   delete m_ctx;
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
   m_lexer->setReader(input, false);
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

   if( isComplete() )
   {
      if( ! m_currentTree->empty() )
      {
         if ( m_currentTree->at(0)->handler()->userFlags() == FALCON_SYNCLASS_ID_AUTOEXPR )
         {
            // this requires evaluation; but is this a direct call?
            StmtAutoexpr* stmt = static_cast<StmtAutoexpr*>(m_currentTree->at(0));
            Expression* expr = stmt->selector();
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
         status= e_definition;
         definition = m_currentMantra;
      }
   }
   else {
      status = e_incomplete;
   }

   return status;
}


void IntCompiler::throwCompileErrors() const
{
   class MyEnumerator: public Parsing::Parser::ErrorEnumerator
   {
   public:
      MyEnumerator() {
         myError = new CodeError( e_compile );
      }

      virtual bool operator()( const Parsing::Parser::ErrorDef& def, bool bLast ){

         String sExtra = def.sExtra;
         if( def.nOpenContext != 0 && def.nLine != def.nOpenContext )
         {
            if( sExtra.size() != 0 )
               sExtra += " -- ";
            sExtra += "from line ";
            sExtra.N(def.nOpenContext);
         }

         SyntaxError* err = new SyntaxError( ErrorParam( def.nCode, def.nLine )
               .origin( ErrorParam::e_orig_compiler )
               .chr( def.nChar )
               .module(def.sUri)
               .extra(sExtra));

         myError->appendSubError(err);

         if( bLast )
         {
            throw myError;
         }

         return true;
      }

   private:
      Error* myError;

   } rator;

   m_sp.enumerateErrors( rator );
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
