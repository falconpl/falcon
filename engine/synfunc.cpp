/*
   FALCON - The Falcon Programming Language.
   FILE: synfunc.h

   SynFunc objects -- expanding to new syntactic trees.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/synfunc.h>
#include <falcon/vmcontext.h>
#include <falcon/item.h>
#include <falcon/trace.h>
#include <falcon/pstep.h>

#include <falcon/psteps/stmtreturn.h>

namespace Falcon
{


SynFunc::SynFunc( const String& name, Module* owner, int32 line ):
   Function( name, owner, line ),
   m_syntree( this ),
   m_bIsPredicate( false )
{
   // by default, use a statement return as fallback cleanup. 
   setPredicate( false );
   m_category = e_c_synfunction;
}


SynFunc::~SynFunc()
{
}


Class* SynFunc::handler() const
{
   static Class* cls = Engine::instance()->synFuncClass();   
   return cls;
}


void SynFunc::setPredicate(bool bmode)
{
   static StmtReturn s_stdReturn;
   
   class PStepReturnRule: public PStep
   {
   public:
      PStepReturnRule() { apply = apply_; }
      virtual ~PStepReturnRule() {}
      void describeTo( String& v, int ) const { v = "Automatic return rule value"; }
      
   private:
      static void apply_( const PStep*, VMContext* ctx ) {
         Item b;
         b.setBoolean( ctx->ruleEntryResult() );
         ctx->returnFrame( b );
      }
   };
   
   static PStepReturnRule s_ruleReturn;
   
   m_bIsPredicate = bmode;
   if( bmode )
   {
      m_retStep = &s_ruleReturn;
   }
   else
   {
      // reset the default return value
      m_retStep = &s_stdReturn;
   }
}


void SynFunc::setConstructor()
{
   class PStepReturnCtor: public PStep
   {
   public:
      PStepReturnCtor() { apply = apply_; }
      virtual ~PStepReturnCtor() {}
      void describeTo( String& v, int ) const { v = "Automatic return constructor value"; }
      
   private:
      static void apply_( const PStep*, VMContext* ctx ) {         
         ctx->returnFrame( ctx->self() );
      }
   };
   
   static PStepReturnCtor s_ctorReturn;
      
    
   // reset the default return value
   m_retStep = &s_ctorReturn;
}

void SynFunc::invoke( VMContext* ctx, int32 nparams )
{  
   register int i;
   
   // nothing to do?
   if( syntree().empty() )
   {
      TRACE( "-- function %s is empty -- not calling it", locate().c_ize() );
      ctx->returnFrame();
      return;
   }
   
   register int paramCount = (int) this->paramCount();
   register int localCount = (int) this->symbols().localCount() - paramCount;
   
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

   // Structure in data stack is:
   // [np0] [np1] [..] [..] [l0] [l1] [..]

   // Add the variables for the parameters.
   Item* itemBase = (&ctx->topData())-nparams;
   for( i = 0; i < paramCount; ++i ) {
      // we're 1 past end; use prefix decrement
      ctx->addLocalVariable(++itemBase);
   }    

   // then add items to be stored in the local variables.
   if( localCount > 0 ) {
      ctx->addLocals( localCount );
      itemBase = (&ctx->topData())-localCount;

      for( i = 0; i < localCount; ++i ) {
         ctx->addLocalVariable(++itemBase);
      }
   }
   
   // push a static return in case of problems.
   ctx->pushCode( m_retStep );
   ctx->pushCode( &this->syntree() );
}

}

/* end of synfunc.cpp */
