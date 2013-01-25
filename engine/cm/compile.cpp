/*
   FALCON - The Falcon Programming Language.
   FILE: compile.cpp

   Falcon core module -- Compile function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 11:12:52 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/compile.cpp"

#include <falcon/cm/compile.h>
#include <falcon/cm/textreader.h>
#include <falcon/cm/textstream.h>
#include <falcon/classes/classstream.h>

#include <falcon/vm.h>
#include <falcon/stringstream.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>
#include <falcon/errors/genericerror.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/sourcelexer.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/syntree.h>
#include <falcon/module.h>

namespace Falcon {
namespace Ext {

   /** Class used to notify the compiler about relevant facts in parsing. */
class FALCON_DYN_CLASS DynCompilerCtx: public ParserContext
{
public:
   DynCompilerCtx(VMContext* ctx, SourceParser& sp):
      ParserContext( &sp ),
      m_ctx(ctx),
      m_sp(sp)
   {}

   virtual ~DynCompilerCtx(){}

   virtual void onInputOver() {}

   virtual Variable* onOpenFunc( Function* ) {
      static Variable var(Variable::e_nt_undefined, Variable::undef, 0, true);
      return &var;
   }

   virtual void onCloseFunc( Function* f) {
      if( f->name() == "") f->name("$anon");
   }

   virtual Variable* onOpenClass( Class*, bool ) {
      static Variable var(Variable::e_nt_undefined, Variable::undef, 0, true);
      return &var;
   }

   virtual void onCloseClass( Class* f, bool ) {
      if( f->name() == "") f->name("$anon");
   }

   virtual void onNewStatement( Statement* ) {}

   virtual void onLoad( const String&, bool ) {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
   }

   virtual bool onImportFrom( ImportDef*  ) {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
      return false;
   }

   virtual void onExport(const String& ) {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
   }
   virtual void onDirective(const String& , const String& ) {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
   }
   virtual void onGlobal( const String&  ) {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
   }

   virtual Variable* onGlobalDefined( const String& , bool&  ) {
      static Variable var(Variable::e_nt_undefined, Variable::undef, 0, true);
      return &var;
   }

   virtual Variable* onGlobalAccessed( const String& ) {
      static Variable var(Variable::e_nt_undefined, Variable::undef, 0, true);
      return &var;
   }

   virtual Item* getVariableValue( const String& name, Variable* ) {
      Item* value = m_ctx->findLocal(name);
      return value;
   }

   virtual void onRequirement( Requirement*  )
   {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
   }

private:
   VMContext* m_ctx;
   SourceParser& m_sp;
};


//============================================================
// The real compiler function
//

Compile::Compile():
   Function( "compile" )
{
   signature("S|Stream|TextReader");
   addParam("code");
}

Compile::~Compile()
{
}

void Compile::invoke( VMContext* ctx , int32 params )
{
   static Class* streamClass = Engine::instance()->streamClass();
   static Class* readerClass = m_module->getClass("TextReader");
   static Class* tsClass = m_module->getClass("TextStream");

   fassert( readerClass!= 0 );
   fassert( tsClass != 0 );

   TextReader* reader = 0;
   Stream* sinput;
   bool ownReader;

   // the parameter can be a string, a stream or a text reader.
   if( params >= 1 ) {
      Item* pitem = ctx->param(0);
      Class* cls;
      void* data;

      if( pitem->isString() ) {
         sinput = new StringStream( *pitem->asString() );
         reader = new TextReader(sinput, true);
         ownReader = true;
      }
      else if( pitem->asClassInst(cls, data) )
      {
         if( cls->isDerivedFrom(streamClass) ) {
            StreamCarrier* sc = static_cast<StreamCarrier*>( data );
            reader = new TextReader( sc->m_underlying, false );
            ownReader = true;
         }
         else if( cls->isDerivedFrom( readerClass ) ) {
            TextReaderCarrier* trc = static_cast<TextReaderCarrier*>( data );
            reader = trc->m_reader;
            ownReader = false;
         }
         else if( cls->isDerivedFrom( tsClass ) ) {
            TextStreamCarrier* tsc = static_cast<TextStreamCarrier*>( data );
            reader = tsc->m_reader;
            ownReader = false;
         }
      }
   }

   if( reader == 0 ) {
      throw paramError(__LINE__, SRC );
   }

   // check the parameters.
   // prepare the parser
   SourceParser sp;
   DynCompilerCtx compctx( ctx, sp );
   sp.setContext( &compctx );


   // start parsing.
   SourceLexer* slex = new SourceLexer( "<internal>", &sp, reader, ownReader );
   SynTree* st = new SynTree;
   sp.pushLexer(slex);

   try
   {
      compctx.openMain( st );

      //TODO: idle the context
      if( ! sp.parse() ) {
         // todo: eventually re-box the error?
         throw sp.makeError();
      }

      ctx->returnFrame( FALCON_GC_HANDLE(st) );
   }
   catch( Error* e )
   {
      delete st;

      ctx->raiseError(e);
      e->decref();
   }
}


}
}

/* end of inspect.cpp */
