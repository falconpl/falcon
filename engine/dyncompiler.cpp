/*
   FALCON - The Falcon Programming Language.
   FILE: dyncompiler.h

   Falcon core module -- Compile function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 11:12:52 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/dymcompiler.cpp"

#include <falcon/dyncompiler.h>
#include <falcon/syntree.h>
#include <falcon/transcoder.h>
#include <falcon/symbol.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/sourcelexer.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/syntree.h>
#include <falcon/module.h>
#include <falcon/attribute_helper.h>
#include <falcon/stringstream.h>
#include <falcon/falconclass.h>


#include <falcon/errors/genericerror.h>

namespace Falcon {

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

   virtual bool onOpenFunc( Function* func ) {
      FALCON_GC_HANDLE( func );
      return true;
   }

   virtual void onCloseFunc( Function* f) {
      if( f->name() == "") f->name("$anon");
   }

   virtual bool onOpenClass( Class* cls, bool isObj ) {
      FALCON_GC_HANDLE( cls );

      if( isObj ) {
         m_sp.addError( e_toplevel_obj, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
         return false;
      }
      return true;
   }

   virtual void onOpenMethod( Class* cls, Function* func) {
      static Variable var(Variable::e_nt_undefined, Variable::undef, 0, true);
      if( static_cast<FalconClass*>(cls)->getProperty(func->name())) {
         m_sp.addError( e_prop_adef, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
      }
   }


   virtual void onCloseClass( Class* f, bool ) {
      if( f->name() == "") f->name("$anon");
   }

   virtual void onNewStatement( TreeStep* ) {}

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

   virtual void onGlobalDefined( const String& , bool&  ) {
   }

   virtual bool onGlobalAccessed( const String& ) {
      return false;
   }

   virtual Item* getValue( Symbol* sym ) {
      Item* value = m_ctx->resolveSymbol( sym, false);
      return value;
   }

   virtual bool onAttribute(const String& name, TreeStep* generator, Mantra* target )
   {
      SourceParser& sp = m_sp;
      if( target == 0 )
      {
         sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      }
      else
      {
         return attribute_helper(m_ctx, name, generator, target );
      }

      return true;
   }

   virtual void onRequirement( Requirement*  )
   {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
   }

   virtual void onIString( const String& )
   {
      // do nothing;
   }

private:
   VMContext* m_ctx;
   SourceParser& m_sp;
};

SynTree* DynCompiler::compile( const String& str, SynTree* target )
{
   static Engine* eng = Engine::instance();

   StringStream ss(str);
   Transcoder* tc;
   switch( str.manipulator()->charSize() )
   {
   case 2: tc = eng->getTranscoder("F16"); break;
   case 4: tc = eng->getTranscoder("F32"); break;
   default: tc = eng->getTranscoder("C"); break;
   }

   return compile(&ss, tc, target);
}

SynTree* DynCompiler::compile( Stream* stream, Transcoder* tc, SynTree* target )
{
   if( tc == 0 )
   {
      LocalRef<TextReader>tr( new TextReader(stream, tc ));
      return compile(&tr, target );
   }
   else {
      LocalRef<TextReader>tr(new TextReader (stream ));
      return compile(&tr, target );
   }
}


SynTree* DynCompiler::compile( TextReader* reader, SynTree* target)
{
   // check the parameters.
   // prepare the parser
   SourceParser sp;
   DynCompilerCtx compctx( m_ctx, sp );
   sp.setContext( &compctx );

   // start parsing.
   SourceLexer* slex = new SourceLexer( "<internal>", &sp, reader );
   SynTree* st = target == 0 ? new SynTree : target;
   sp.pushLexer(slex);

   try
   {
      compctx.openMain( st );

      //TODO: idle the context
      if( ! sp.parse() ) {
         // todo: eventually re-box the error?
         throw sp.makeError();
      }

      return st;
   }
   catch( Error* e )
   {
      if( target == 0 )
      {
         delete st;
      }

      throw e;
   }
   return 0;
}

}

/* end of dyncompiler.cpp */
