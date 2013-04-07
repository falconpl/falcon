/*
   FALCON - The Falcon Programming Language.
   FILE: stmtwhile.cpp

   Statatement -- while
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:51:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/psteps/stmtwhile.cpp"

#include <falcon/trace.h>
#include <falcon/psteps/stmtwhile.h>
#include <falcon/expression.h>
#include <falcon/vmcontext.h>
#include <falcon/syntree.h>
#include <falcon/textwriter.h>

#include <falcon/engine.h>
#include <falcon/synclasses.h>

namespace Falcon
{

StmtWhile::StmtWhile(int32 line, int32 chr ):
   Statement( line, chr ),
   m_child( 0 ),
   m_expr( 0 )
{
   FALCON_DECLARE_SYN_CLASS( stmt_while );
   
   apply = apply_;
   m_bIsLoopBase = true;
   m_bIsNextBase = true;
}

StmtWhile::StmtWhile( Expression* expr, TreeStep* stmts, int32 line, int32 chr ):
   Statement( line, chr ),
   m_child( stmts ),
   m_expr( expr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_while );
   
   stmts->setParent(this);
   expr->setParent(this);
   
   apply = apply_;
   m_bIsLoopBase = true;
   m_bIsNextBase = true;
}


StmtWhile::StmtWhile( Expression* expr, int32 line, int32 chr ):
   Statement( line, chr ),
   m_child( 0 ),
   m_expr( expr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_while );
   
   expr->setParent(this);
   apply = apply_;
   m_bIsLoopBase = true;
   m_bIsNextBase = true;
}



StmtWhile::StmtWhile( const StmtWhile& other ):
   Statement( other ),
   m_child( 0 ),
   m_expr( 0 )
{
   apply = apply_;
   m_bIsLoopBase = true;
   m_bIsNextBase = true;
   
   if( other.m_child )
   {
      m_child = other.m_child->clone();
      m_child->setParent(this);
   }
   
   if( other.m_expr != 0 )
   {
      m_expr = other.m_expr->clone();
      m_expr->setParent(this);      
   }
}

StmtWhile::~StmtWhile()
{
   dispose( m_child );
   dispose( m_expr );
}


void StmtWhile::minimize()
{
   m_child = minimize_basic(m_child);
}

TreeStep* StmtWhile::selector() const
{
   return m_expr;
}


bool StmtWhile::selector( TreeStep* e )
{
   if( e!= 0 && e->setParent(this))
   {
      dispose( m_expr );
      m_expr = e;
      return true;
   }
   return false;
}


void StmtWhile::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   if( m_expr == 0 )
   {
      tw->write( "/* Blank StmtWhile */" );
   }
   else
   {
      tw->write( "while " );
      m_expr->render( tw, relativeDepth(depth) );
      if( m_child != 0 )
      {
         m_child->render(tw, depth < 0 ? -depth : depth + 1 );
      }
   }
   tw->write( renderPrefix(depth) );
   tw->write( "end" );
   if( depth >=  0 )
   {
      tw->write( "\n" );
   }
}


int StmtWhile::arity() const
{
   if ( m_child == 0 )
   {
      return 0;
   }

   if( m_child->category() == TreeStep::e_cat_syntree )
   {
      return m_child->arity();
   }

   // a single non-syntree child
   return 1;
}

TreeStep* StmtWhile::nth( int n ) const
{
   if( m_child ==  0)
   {
      return 0;
   }

   if( m_child->category() == TreeStep::e_cat_syntree )
   {
      return m_child->nth(n);
   }

   // a single child
   if( n == 0 || n == -1 ) return m_child;
   return 0;
}


bool StmtWhile::setNth( int n, TreeStep* st )
{
   if( n == arity() )
   {
      return append(st);
   }

   if( m_child == 0 )
   {
      // if we're here, n > 1, as arity without a child is 0
      return false;
   }


   if( m_child->category() == TreeStep::e_cat_syntree )
   {
      return m_child->setNth(n, st);
   }

   // single child
   if( st == 0 || (n != 0 && n != -1) || ! st->setParent(this)  ) return false;

   dispose( m_child );
   m_child = st;

   return true;
}


bool StmtWhile::insert( int32 pos, TreeStep* element )
{
   // IMPORTANT: we can't set the parent prior being certain the insertion position is valid.
   if( m_child == 0 )
   {
      if( pos == 0 && element->setParent(this) )
      {
         m_child = element;
         return true;
      }
      return false;
   }

   if( m_child->category() == TreeStep::e_cat_syntree )
   {
      return m_child->insert(pos, element);
   }

   // a single child.
   if( (pos == 0 || pos == -1) && element->parent() == 0 )
   {
      // disengage the current child.
      singleToMultipleChild( element, false );
   }

   return false;
}


bool StmtWhile::append( TreeStep* element )
{
   if( element->parent() != 0 )
   {
      return false;
   }

   if( m_child == 0 )
   {
      m_child = element;
      element->setParent(this);
   }
   else if( m_child->category() == SynTree::e_cat_syntree )
   {
      m_child->append( element );
   }
   else {
      // disengage the current child.
      singleToMultipleChild( element, true );
   }

   return true;
}


void StmtWhile::singleToMultipleChild( TreeStep* element, bool last )
{
   TreeStep* child = m_child;
   child->setParent(0);
   m_child = new SynTree;
   m_child->setParent(this);

   if( last )
   {
      m_child->append( child );
      m_child->append( element );
   }
   else {
      m_child->append( element );
      m_child->append( child );
   }
}

void StmtWhile::mainBlock(TreeStep* st) {
   dispose( m_child );
   st->setParent(this);
   m_child = st;
}

TreeStep* StmtWhile::detachMainBlock()
{
   m_child->setParent(0);
   TreeStep* ret = m_child;
   m_child = 0;
   return ret;
}


void StmtWhile::apply_( const PStep* s1, VMContext* ctx )
{
   const StmtWhile* self = static_cast<const StmtWhile*>(s1);
   TRACE( "StmtWhile::apply_ entering %s", self->describe().c_ize() );
   fassert( self->m_expr != 0 );
   
   CodeFrame& cf = ctx->currentCode();
   
   // Perform the check
   TreeStep* tree = self->m_child;
   switch ( cf.m_seqId )
   {
   case 0:
      // preprare the stack
      ctx->saveUnrollPoint( cf );

      // generate the first expression
      cf.m_seqId = 2;
      if( ctx->stepInYield( self->m_expr, cf ) )
      {
          // ignore soft exception, we're yielding back soon anyhow.
          return;
      }
      break;

   case 1:
      // already been around
      ctx->popData(); // remove the data placed by the syntree

      cf.m_seqId = 2;
      if( ctx->stepInYield( self->m_expr, cf ) )
      {
          // ignore soft exception, we're yielding back soon anyhow.
          return;
      }
      break;
   }
   

   // break items are always nil, and so, false.
   if ( ctx->boolTopData() )
   {
      ctx->popData();
      // mark for regeneration of the expression
      cf.m_seqId = 1;
      TRACE1( "Apply 'while' at line %d -- redo", self->line() );
      // redo
      if( tree != 0 ) {
         ctx->stepIn( tree );
      }
      else {
         ctx->pushData(Item());
         return;
      }
      // no matter if stmts went deep, we're bound to be called again to recheck
   }
   else {      
      TRACE1( "Apply 'while' at line %d -- leave ", self->line() );
      //we're done
      //keep the data
      ctx->topData().setNil();
      ctx->popCode();
   }
}
         
}

/* end of stmtwhile.cpp */
