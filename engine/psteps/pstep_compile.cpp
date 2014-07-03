/*
   FALCON - The Falcon Programming Language.
   FILE: pstep_compile.h

   A step continuously invoking and evaluating a compilation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 21 Nov 2012 14:20:53 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/psteps/breakpoint.cpp"
#include <falcon/psteps/pstep_compile.h>
#include <falcon/textreader.h>
#include <falcon/textwriter.h>
#include <falcon/stream.h>
#include <falcon/error.h>
#include <falcon/trace.h>

namespace Falcon {

class PStepCompile::CatchSyntree: public SynTree
{
public:
   CatchSyntree( PStepCompile* ps )
   {
      apply = apply_;
      m_compiler = ps;
   }

private:
   static void apply_( const PStep* self, VMContext* ctx ){
      MESSAGE( "CatchSyntree::apply_");
      const CatchSyntree* cst = static_cast<const CatchSyntree*>(self);

      // put our parent back in charge.
      ctx->resetCode( cst->m_compiler );
      if( ctx->thrownError() == 0 )
      {
         UncaughtError* error = FALCON_SIGN_XERROR(UncaughtError, e_uncaught, .extra("Interactive"));
         error->raised( ctx->raised() );
         cst->m_compiler->onError( error );
         error->decref();
      }
      else {
         cst->m_compiler->onError(ctx->thrownError());
      }
   }

   virtual void describeTo( String& tgt ) const
   {
        //Need to do something about this
      tgt = String(" ").replicate( 0 ) + "/*CatchSyntree*/";
   }

   virtual void oneLinerTo( String& tgt ) const
   {
      tgt = "/*CatchSyntree*/";
   }

   PStepCompile* m_compiler;
};

PStepCompile::PStepCompile(int32 line, int32 chr):
         StmtTry( line, chr ),
         m_function( 0 ),
         m_module( 0 )
{
   // This is not a syntactic pstep, we don't need to have a syntax for this.
   apply = apply_;
   m_compiler = new IntCompiler;
   catchSelect().append( new CatchSyntree(this) );
}

PStepCompile::PStepCompile( const PStepCompile& other ):
         StmtTry( other ),
         m_function( 0 ),
         m_module( 0 )
{
   apply = apply_;
   m_compiler = new IntCompiler;
   catchSelect().append( new CatchSyntree(this) );
}

PStepCompile::~PStepCompile()
{
   delete m_compiler;
}

PStepCompile* PStepCompile::clone() const
{
   return new PStepCompile(*this);
}

void PStepCompile::describeTo( String& tgt ) const
{
   tgt ="<compile>";
}

void PStepCompile::setCompilerContext( Function* func, Module* mod, TextReader* tin, TextWriter* tout )
{
   m_function = func;
   m_module = mod;

   m_tin = tin;
   m_tout = tout;
}


void PStepCompile::apply_( const PStep* ps, VMContext* ctx )
{
   const PStepCompile* psc = static_cast<const PStepCompile*>(ps);
   IntCompiler* comp = psc->m_compiler;
   comp->setCompilationContext( psc->m_function, psc->m_module, ctx );

   CodeFrame& curcode = ctx->currentCode();
   int status = ctx->currentCode().m_seqId;
   ctx->currentCode().m_seqId = 0; // always reset prior next operation.
   String tgt;
   String prompt = comp->isComplete() ? ">>> " : "... ";

   // print result if needed.
   if ( status == 2 || (status == 1 && ! ctx->topData().isNil()) ) {
      Class* cls = 0;
      void* inst = 0;
      ctx->topData().forceClassInst(cls, inst);
      String result;
      cls->describe( inst, result, 3, 60 );
      psc->m_tout->write( result + "\n" );
   }

   if( status > 0 ) {
      ctx->popData();
   }

   while( true )
   {
      psc->m_tout->write( prompt );
      psc->m_tout->flush();
      //psc->m_tin->underlying()->readAvailable(-1);

      try {
         SynTree* st = 0;
         Mantra* decl = 0;
         IntCompiler::t_compile_status status = comp->compileNext( psc->m_tin, st, decl );

         if( st != 0 )
         {
            // declare how we shall print the result on return.
            if( status == IntCompiler::e_expression ) {
               ctx->currentCode().m_seqId = 2;
            }
            else if( status == IntCompiler::e_expr_call ) {
               ctx->currentCode().m_seqId = 1;
            }
            else {
               // indicate we're already called.
               ctx->currentCode().m_seqId = 3;
            }

            ctx->pushCode( st );
            break;
         }
         else if( psc->m_tin->eof() ) {

            // we're done.
            ctx->popCode();
            return;
         }
      }
      catch( Error* e )
      {
         // this catches compilation error codes only.
         psc->onError( e );
         e->decref();
      }

      if( &ctx->currentCode() != &curcode )
      {
         // went deep? -- go away, we'll be back
         break;
      }

      // resets the prompt
      prompt = comp->isComplete() ? ">>> " : "... ";
   }
}

void PStepCompile::onError( Error* e ) const
{
   // display the error and continue
   if( e->errorCode() == e_compile )
   {
      // in case of a compilation, discard the encapsulator.
      class MyEnumerator: public Error::ErrorEnumerator {
      public:
         MyEnumerator( TextWriter* wr ):
            m_wr(wr)
         {}

         virtual bool operator()( const Error& e ){
            m_wr->write(e.describe(false)+"\n");
            return true;
         }
      private:
         TextWriter* m_wr;
      } rator(this->m_tout);

      e->enumerateErrors( rator );
   }
   else {
      this->m_tout->write(e->describe()+"\n");
   }
}

}

/* end of pstep_compile.h */
