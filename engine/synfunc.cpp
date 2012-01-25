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
#include <falcon/itemarray.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>

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

// As this is called by the engne at init, we don't have to care much about
// mt clashes and so on.
MetaStorer* SynFunc::metaStorer() const
{
   static MetaStorer* theStorer = Engine::instance()->getMetaStorer("$.SynFunc");
   fassert( theStorer != 0 );
   
   return theStorer;
}


//==========================================================================
// Storer for synfunctions.
//

SynFunc::SynStorer::SynStorer():
   MetaStorer( "$.SynFunc" )
{}

void SynFunc::SynStorer::store( VMContext*, DataWriter* stream, void* instance ) const
{
   SynFunc* synfunc = static_cast<SynFunc*>(instance);
   // saving the overall data.
   stream->write( synfunc->name() );
   stream->write( synfunc->declaredAt() );
   stream->write( synfunc->isPredicate() );
   stream->write( synfunc->isEta() );
   stream->write( synfunc->signature() );
   
   // now we got to save the function parameter table.
   stream->write( synfunc->paramCount() );
   synfunc->symbols().store( stream );
}

void SynFunc::SynStorer::restore( VMContext*, DataReader* stream, void*& empty ) const
{
   bool bPred, bEta;
   int line;
   String name, signature;
   
   stream->read( name );
   stream->read( line );
   stream->read( bPred );
   stream->read( bEta );
   stream->read( signature );
   
   
   SynFunc* synfunc = new SynFunc( name, 0, line );
   synfunc->setPredicate( bPred );
   synfunc->setEta( bEta );
   synfunc->signature( signature );
   
   int32 pcount;
   stream->read(pcount);
   synfunc->paramCount( pcount );
   synfunc->symbols().restore( stream );
   
   empty = synfunc;
}

void SynFunc::SynStorer::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   SynFunc* synfunc = static_cast<SynFunc*>(instance);
   subItems.reserve(synfunc->syntree().size());
   for( uint32 i = 0; i < synfunc->syntree().size(); ++i ) {
      Statement* stmt = synfunc->syntree().at(i);
      Class* synClass = stmt->cls();
      subItems.append(Item( synClass, stmt ) );
   }
}

void SynFunc::SynStorer::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{    
   SynFunc* synfunc = static_cast<SynFunc*>(instance);
   
   for( uint32 i = 0; i < subItems.length(); ++i ) {
      Class* cls = 0;
      void* data = 0;
      subItems[i].asClassInst(cls,data);
      
#ifndef NDEBUG
      static Class* stmtClass = Engine::instance()->statementClass();
      fassert2( cls != 0, "Serialized instances are not classes" );
      fassert2( cls->isDerivedFrom( stmtClass ), "Serialized instances are not statements" );               
#endif
      
      synfunc->syntree().append( static_cast<Statement*>(data) );
   }
}

}

/* end of synfunc.cpp */
