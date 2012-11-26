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

namespace Falcon {

PStepCompile::PStepCompile(int32 line, int32 chr):
         PStep( line, chr ),
         m_function( 0 ),
         m_module( 0 )
{
   // This is not a syntactic pstep, we don't need to have a syntax for this.
   apply = apply_;
   m_compiler = new IntCompiler;
   m_bIsCatch = true;
}

PStepCompile::PStepCompile( const PStepCompile& other ):
         PStep( other ),
         m_function( 0 ),
         m_module( 0 )
{
   apply = apply_;
   m_compiler = new IntCompiler;
   m_bIsCatch = true;
}

PStepCompile::~PStepCompile()
{
   delete m_compiler;
}

PStepCompile* PStepCompile::clone() const
{
   return new PStepCompile(*this);
}

void PStepCompile::describeTo( String& tgt, int ) const
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

   String tgt;
   String prompt = ">>> ";

   while( true )
   {
      psc->m_tout->write( prompt );
      psc->m_tout->flush();


      try {
         SynTree* st = 0;
         Mantra* decl = 0;
         /*IntCompiler::t_compile_status status = */ comp->compileNext( psc->m_tin, st, decl );

         if( st != 0 )
         {
            // todo: delete the steps
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
         // display the error and continue
         if( e->errorCode() == e_compile )
         {
            // in case of a compilation, discard the encapsulator.
            class MyEnumerator: public Error::ErrorEnumerator {
            public:
               MyEnumerator( TextWriter* wr ):
                  m_wr(wr)
               {}

               virtual bool operator()( const Error& e, bool  ){
                  m_wr->write(e.describe()+"\n");
                  return true;
               }
            private:
               TextWriter* m_wr;
            } rator(psc->m_tout);

            e->enumerateErrors( rator );
         }
         else {
            psc->m_tout->write(e->describe()+"\n");
         }

         e->decref();
      }

      // resets the prompt
      prompt = comp->isComplete() ? ">>> " : "... ";
   }

}

}

/* end of pstep_compile.h */
