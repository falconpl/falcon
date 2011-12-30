/*
   FALCON - The Falcon Programming Language.
   FILE: stmttry.cpp

   Syntactic tree item definitions -- Try/catch.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 15 Aug 2011 13:58:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/stmttry.cpp"

#include <falcon/syntree.h>
#include <falcon/vmcontext.h>

#include <falcon/psteps/stmttry.h>

#include <falcon/engine.h>
#include <falcon/synclasses.h>

namespace Falcon {

StmtTry::StmtTry( int32 line, int32 chr):
   Statement( line, chr ),
   m_body( new SynTree ),
   m_fbody( 0 ),
   m_finallyStep( this )
{
   FALCON_DECLARE_SYN_CLASS( stmt_try );
   
   m_select.setParent(this);
   apply = apply_;
   m_bIsCatch = true;
}

StmtTry::StmtTry( SynTree* body, int32 line, int32 chr ):
   Statement( line, chr ),
   m_body( body ),
   m_fbody( 0 ),
   m_finallyStep( this )
{
   FALCON_DECLARE_SYN_CLASS( stmt_try );

   m_select.setParent(this);
   body->setParent( this );
   apply = apply_;
   m_bIsCatch = true;
}

StmtTry::StmtTry( SynTree* body, SynTree* fbody, int32 line, int32 chr ):
   Statement( line, chr ),
   m_body( body ),
   m_fbody( fbody ),
   m_finallyStep( this )
{
   FALCON_DECLARE_SYN_CLASS( stmt_try );

   m_select.setParent(this);
   body->setParent( this );
   fbody->setParent(this);
   apply = apply_;
   m_bIsCatch = true;
}

StmtTry::StmtTry( const StmtTry& other ):
   Statement( other ),
   m_body( 0 ),
   m_fbody( 0 ),
   m_select( other.m_select ),
   m_finallyStep( this )
{
   apply = apply_;   
   m_select.setParent(this);
   
   if( other.m_body )
   {
      m_body = other.m_body->clone();
      m_body->setParent( this );
      m_bIsCatch = true;
   }
   
   if( other.m_fbody )
   {
      m_fbody = other.m_fbody->clone();
      m_fbody->setParent( this );
   }
}


StmtTry::~StmtTry()
{
   delete m_body;
   delete m_fbody;
}


bool StmtTry::body( SynTree* body )
{ 
   if( body->setParent(this) )
   {
      delete m_body;
      m_body = body;
      return true;
   }
   return false;
}


bool StmtTry::fbody( SynTree* body )
{ 
   if( body->setParent(this) )
   {
      delete m_fbody;
      m_fbody = body;
      return true;
   }
   return false;
}

void StmtTry::describeTo( String& tgt, int depth ) const
{
   // TODO: describe catches, finally & check various options.
   if( m_body == 0 )
   {
      tgt = "<Blank StmtTry>";
      return;
   }
   
   String prefix = String(" ").replicate( depth * depthIndent );
   tgt = prefix + "try\n" + m_body->describe( depth + 1 ) + "\n" + prefix + "end";   
}


void StmtTry::oneLinerTo( String& tgt ) const
{
   tgt = "try ...";
}

   
void StmtTry::apply_( const PStep* ps, VMContext* ctx )
{ 
   const StmtTry* self = static_cast<const StmtTry*>(ps);   
   
   if( self->m_body != 0 )
   {
      // push the finally step?
      if( self->m_fbody != 0 )
      {
         // put the finally processor in our place...
         ctx->resetCode( &self->m_finallyStep );
         // add a catch position placeholder
         ctx->pushCode( &self->m_stepDone );
         // ... and declare that we'll be doing some finally
         ctx->traverseFinally();
      }
      else
      {
         // just add the catch placeholder.
         ctx->resetCode( &self->m_stepDone );
      }
      
      if( ctx->stepInYield( self->m_body ) )
         return;
   }
   else if( self->m_fbody )
   {
      // we're just doing the finally!
      ctx->popCode();
      ctx->stepIn( self->m_fbody );
   }   
}

void StmtTry::PStepDone::apply_( const PStep*, VMContext* ctx )
{ 
   // we're just a placeholder for our catch clauses,
   // if we're here, then we had no throws.
   ctx->popCode(); 
}


void StmtTry::PStepFinally::apply_( const PStep* ps, VMContext* ctx )
{
   register const StmtTry* stry = static_cast<const StmtTry::PStepFinally*>(ps)->m_owner;
   
   // declare that we begin to work with finally
   ctx->enterFinally();
   ctx->currentCode().m_step = &stry->m_cleanStep;
   ctx->stepIn( stry->m_fbody );   
}


void StmtTry::PStepCleanup::apply_( const PStep*, VMContext* ctx )
{
   // declare that we're ready to be completed
   ctx->popCode();
   ctx->finallyComplete();
}


}

/* end of stmttry.cpp */
