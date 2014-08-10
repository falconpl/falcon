/*
   FALCON - The Falcon Programming Language.
   FILE: synfunc.cpp

   SynFunc objects -- expanding to new syntactic trees.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/synfunc.cpp"

#include <falcon/synfunc.h>
#include <falcon/vmcontext.h>
#include <falcon/item.h>
#include <falcon/trace.h>
#include <falcon/pstep.h>
#include <falcon/stdsteps.h>
#include <falcon/stdhandlers.h>
#include <falcon/textwriter.h>
#include <falcon/attribute.h>

namespace Falcon
{


SynFunc::SynFunc( const String& name, Module* owner, int32 line ):
   Function( name, owner, line ),
   m_syntree( this )
{
   static PStep* retStep = &Engine::instance()->stdSteps()->m_returnFrame;

   m_category = e_c_synfunction;
   m_bIsConstructor = false;
   m_retStep = retStep;
}


SynFunc::~SynFunc()
{
}


const Class* SynFunc::handler() const
{
   static const Class* cls = Engine::handlers()->synFuncClass();
   return cls;
}

void SynFunc::setConstructor()
{
   class PStepReturnCtor: public PStep
   {
   public:
      PStepReturnCtor() { apply = apply_; }
      virtual ~PStepReturnCtor() {}
      void describeTo( String& v ) const { v = "Automatic return constructor value"; }

   private:
      static void apply_( const PStep*, VMContext* ctx ) {
         ctx->returnFrame( ctx->self() );
      }
   };

   static PStepReturnCtor s_ctorReturn;

   m_bIsConstructor = true;
   // reset the default return value
   m_retStep = &s_ctorReturn;
}


#if 0
void SynFunc::invoke( VMContext* ctx, int32 nparams )
{
   // nothing to do?
   register int paramCount = (int) this->paramCount();
   register int localCount = (int) this->parameters().localCount();

   // fill the parameters
   TRACE1( "-- filing parameters: %d/%d, and locals %d",
         nparams, paramCount,
         localCount );

   // fill the named parameters if not enough
   if( paramCount > nparams )
   {
      ctx->addLocals( paramCount - nparams );
      // equalized the passed parameters.
      nparams = paramCount;
   }

   //ctx->addLocals( localCount );
   // Structure in data stack is:
   // [np0] [np1] [..] [..] [l0] [l1] [..]

   // push a static return in case of problems.
   ctx->pushCode( m_retStep );
   ctx->pushCode( &this->syntree() );
}
#endif

void SynFunc::invoke( VMContext* ctx, int32 nparams )
{
   // nothing to do?
   register int paramCount = (int) this->paramCount();

   // fill the parameters
   TRACE1( "-- filing parameters: %d/%d", nparams, paramCount );

   // ok even if nparams = 0; we won't use base
   Item* base = ctx->opcodeParams(nparams);

   // fill the named parameters if not enough
   int32 filledParams = paramCount < nparams ? paramCount : nparams;
   int32 i = 0;
   const Symbol* paramSym;

   while( i < filledParams )
   {
      paramSym = this->parameters().getById(i);
      fassert( paramSym != 0 );
      ctx->defineSymbol( paramSym, base );
      ++i;
      ++base;
   }

   while( i < paramCount )
   {
      paramSym = this->parameters().getById(i);
      ctx->defineSymbol( paramSym );
      ++i;
   }


   // push a static return in case of problems.
   ctx->pushCode( m_retStep );
   ctx->pushCode( &this->syntree() );
}

void SynFunc::renderFunctionBody( TextWriter* tgt, int32 depth ) const
{
   attributes().render(tgt, depth);
   syntree().render( tgt, depth );
}

}

/* end of synfunc.cpp */
