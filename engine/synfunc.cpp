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
}


SynFunc::~SynFunc()
{
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


void SynFunc::invoke( VMContext* ctx, int32 nparams )
{   
   // nothing to do?
   if( syntree().empty() )
   {
      TRACE( "-- function %s is empty -- not calling it", locate().c_ize() );
      ctx->returnFrame();
      return;
   }
   
   // fill the parameters
   TRACE1( "-- filing parameters: %d/%d, and locals %d",
         nparams, this->paramCount(),
         this->symbols().localCount() - this->paramCount() );

   register int lc = (int) this->symbols().localCount();
   if( lc > nparams )
   {
      ctx->addLocals( lc - nparams );
   }
   
   // push a static return in case of problems.   
   ctx->pushCode( m_retStep );
   ctx->pushCode( &this->syntree() );
}

}

/* end of synfunc.cpp */
