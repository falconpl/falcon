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
#include <falcon/textwriter.h>
#include <falcon/symbol.h>
#include <falcon/expression.h>

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
   
   m_select = new StmtSelect;
   m_select->setParent(this);
   apply = apply_;
   setTry();
}

StmtTry::StmtTry( SynTree* body, int32 line, int32 chr ):
   Statement( line, chr ),
   m_body( body ),
   m_fbody( 0 ),
   m_finallyStep( this )
{
   FALCON_DECLARE_SYN_CLASS( stmt_try );

   m_select = new StmtSelect;
   body->setParent( this );
   apply = apply_;
   setTry();
}

StmtTry::StmtTry( SynTree* body, SynTree* fbody, int32 line, int32 chr ):
   Statement( line, chr ),
   m_body( body ),
   m_fbody( fbody ),
   m_finallyStep( this )
{
   FALCON_DECLARE_SYN_CLASS( stmt_try );

   m_select = new StmtSelect;
   m_select->setParent(this);
   body->setParent( this );
   fbody->setParent(this);
   apply = apply_;
   setTry();
}

StmtTry::StmtTry( const StmtTry& other ):
   Statement( other ),
   m_body( 0 ),
   m_fbody( 0 ),
   m_select( new StmtSelect(*other.m_select) ),
   m_finallyStep( this )
{
   setTry();
   apply = apply_;   
   m_select->setParent(this);
   
   if( other.m_body )
   {
      m_body = other.m_body->clone();
      m_body->setParent( this );
   }
   
   if( other.m_fbody )
   {
      m_fbody = other.m_fbody->clone();
      m_fbody->setParent( this );
   }
}


StmtTry::~StmtTry()
{
   dispose( m_body );
   dispose( m_fbody );
   dispose( m_select );
}


bool StmtTry::body( SynTree* body )
{ 
   if( body->setParent(this) )
   {
      dispose( m_body );
      m_body = body;
      return true;
   }
   return false;
}


bool StmtTry::fbody( SynTree* body )
{ 
   if( body->setParent(this) )
   {
      dispose( m_fbody );
      m_fbody = body;
      return true;
   }
   return false;
}

int32 StmtTry::arity() const
{
   return 3;
}


TreeStep* StmtTry::nth( int32 n ) const
{
   switch( n ) {
   case 0: case -3: return m_body;
   case 1: case -2: return m_select;
   case 2: case -1: return m_fbody;
   }

   return 0;
}


bool StmtTry::setNth( int32 n, TreeStep* ts )
{
   static Class* slc = Engine::instance()->synclasses()->m_stmt_select;

   switch( n ) {
   case 0: case -3:
      if ( ts->category() == TreeStep::e_cat_syntree ) {
         SynTree* st = static_cast<SynTree*>( ts );
         return body(st);
      }
      break;

   case 1: case -2:
         if ( ts->handler() == slc ) {
            StmtSelect* st = static_cast<StmtSelect*>( ts );
            if( st->setParent(this) )
            {
               dispose( m_select );
               m_select = st;
               return true;
            }
         }
         break;

   case 2: case -1:
      if ( ts->category() == TreeStep::e_cat_syntree ) {
         SynTree* st = static_cast<SynTree*>( ts );
         return fbody(st);
      }
      break;
   }

   return false;
}


void StmtTry::render( TextWriter* tw, int32 depth ) const
{

   int32 dp = depth < 0 ? -depth : depth+1;
   tw->write( renderPrefix(depth) );
   tw->write( "try\n" );

   if( m_body != 0 )
   {
      m_body->render(tw, dp);
   }
   
   if( m_select != 0 )
   {
      int32 count = m_select->arity();
      for( int32 i = 0; i < count; ++i )
      {
         SynTree* handler = static_cast<SynTree*>(m_select->nth(i));
         if( handler->category() != SynTree::e_cat_syntree )
         {
            continue;
         }

         tw->write( renderPrefix(depth) );
         tw->write( "catch " );

         if( handler->selector() != 0 )
         {
            handler->selector()->render(tw, relativeDepth(dp) );
         }

         if( handler->target() != 0 )
         {
            if( handler->isTracedCatch() )
            {
               tw->write( " as " );
            }
            else {
               tw->write( " in ");
            }
            tw->write( handler->target()->name() );
         }
         tw->write( "\n" );

         handler->render( tw, dp );

         if( handler->selector() == 0 )
         {
            // force catch-all to be the last
            break;
         }
      }
   }

   if( m_fbody != 0 )
   {
      tw->write( renderPrefix(depth) );
      tw->write( "finally\n" );
      m_fbody->render(tw, dp );
   }

   tw->write( renderPrefix(depth) );
   tw->write("end");

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}
   
void StmtTry::apply_( const PStep* ps, VMContext* ctx )
{ 
   const StmtTry* self = static_cast<const StmtTry*>(ps);   

   // first time around?
   CodeFrame& cf = ctx->currentCode();
   TRACE( "StmtTry::apply_ %d/1 %s ", cf.m_seqId, self->describe().c_ize() );
   if( cf.m_seqId == 0 )
   {
      // preliminary checks.
      if( self->m_body == 0 )
      {
         if( self->m_fbody == 0 )
         {
            MESSAGE( "StmtTry::apply_ Removed because fully empty");
            ctx->popCode();
            return;
         }
         else {
            MESSAGE( "StmtTry::apply_ substituting with finally");
            ctx->resetCode( self->m_fbody );
            self->m_fbody->apply( self->m_fbody, ctx );
            return;
         }
      }

      // If we have a finally...
      if( self->m_fbody != 0 )
      {
         // change this step into a finally gate
         ctx->resetCode( &self->m_finallyStep );
         ctx->saveUnrollPoint(cf);

         // save the finally point
         ctx->registerFinally(self->m_fbody);

         // now repush us as a catch point
         ctx->pushCodeWithUnrollPoint(self);
         ctx->currentCode().m_seqId = 1;
      }
      else {
         // keep us, but change into a catch point
         cf.m_seqId = 1;
         ctx->saveUnrollPoint(cf);
      }

      // Push in the body
      ctx->pushCode( self->m_body );
   }
   else {
      // we're off
      ctx->popCode();
   }
}


void StmtTry::PStepFinally::apply_( const PStep* ps, VMContext* ctx )
{
   register const StmtTry* stry = static_cast<const StmtTry::PStepFinally*>(ps)->m_owner;
   // pop the topomost finally barrier
   ctx->unregisterFinally();
   ctx->resetCode( stry->m_fbody );
   stry->m_fbody->apply(stry->m_fbody, ctx);
}

}

/* end of stmttry.cpp */
