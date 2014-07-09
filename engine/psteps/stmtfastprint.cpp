/*
   FALCON - The Falcon Programming Language.
   FILE: stmtfastprint.cpp

   Syntactic tree item definitions -- Fast Print statement.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 14 Aug 2011 12:41:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/stmtfastprint.cpp"


#include <falcon/expression.h>
#include <falcon/vm.h>
#include <falcon/textwriter.h>
#include <falcon/psteps/stmtfastprint.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include "exprvector_private.h"

namespace Falcon
{

Mutex StmtFastPrint::m_mtx;

class StmtFastPrint::Private: public TSVector_Private<Expression>
{
public:

   Private() {}
   ~Private() {}

   Private( const Private& other, TreeStep* owner ):
      TSVector_Private<Expression>( other, owner )
   {}
};


StmtFastPrint::StmtFastPrint( int line, int chr ):
   Statement( line, chr ),
   _p( new Private ),
   m_bAddNL( false )
{
   FALCON_DECLARE_SYN_CLASS(stmt_fastprint)
   apply = apply_;
}


StmtFastPrint::StmtFastPrint( int line, int chr, bool ):
   Statement( line, chr ),
   _p( new Private ),
   m_bAddNL( false )
{
   // do not declare the synclass, this constructor is for child classes
   apply = apply_;
}

StmtFastPrint::StmtFastPrint( const StmtFastPrint& other ):
   Statement( other ),
   _p( new Private(*other._p, this) ),
   m_bAddNL( other.m_bAddNL )
{
   apply = apply_;
}

StmtFastPrint::~StmtFastPrint()
{
   delete _p;
}

int StmtFastPrint::arity() const
{
   return _p->arity();
}

TreeStep* StmtFastPrint::nth( int32 n ) const
{
   return _p->nth(n);
}

bool StmtFastPrint::setNth( int32 n, TreeStep* ts )
{
   if( ts == 0
      || ts->category() != TreeStep::e_cat_expression )
   {
      return false;
   }
   return _p->nth(n, static_cast<Expression*>(ts), this );
}

bool StmtFastPrint::insert( int32 n, TreeStep* ts )
{
   if( ts == 0
      || ts->category() != TreeStep::e_cat_expression )
   {
      return false;
   }

   return _p->insert(n, static_cast<Expression*>(ts), this );
}

bool StmtFastPrint::remove( int32 n )
{
   return _p->remove(n);
}


void StmtFastPrint::add( Expression* expr )
{
   if( expr->setParent(this) )
   {
      _p->m_exprs.push_back( expr );
   }
}


Expression* StmtFastPrint::at( int n ) const
{
   return _p->m_exprs[ n ];
}


length_t StmtFastPrint::size() const
{
   return _p->m_exprs.size();
}

void StmtFastPrint::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   tw->write( m_bAddNL ? "> " : ">> " );
   Private::ExprVector::iterator iter = _p->m_exprs.begin();
   while( iter != _p->m_exprs.end() )
   {
      (*iter)->render( tw, relativeDepth(depth) );
      ++iter;
      if( iter != _p->m_exprs.end() )
      {
         tw->write( ", " );
      }
   }

   if( depth >= 0 )
   {
      tw->write( "\n" );
   }
}


void StmtFastPrint::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtFastPrint* self = static_cast< const StmtFastPrint*>( ps );
   self->m_mtx.lock();
   Private::ExprVector& pl = self->_p->m_exprs;
   CodeFrame& cframe = ctx->currentCode();
   int seqId = cframe.m_seqId;

   // can we print?
   if( seqId > 0 )
   {
      register Item& top = ctx->topData();

      if( top.isString() )
      {
         ctx->process()->textOut()->write( *top.asString() );
      }
      else
      {
         // complex item?
         Class* cls; void* data;
         if( top.asClassInst( cls, data ) )
         {
            // stringify
            cls->op_toString( ctx, data );
            // went deep?
            if( &cframe != &ctx->currentCode() )
            {
               // we'll perform the same check again when the deep op is done,
               // hopefully, it will have transformed the data into a string.
               return;
            }

            // else we can proceed here.
            if( top.isString() )
            {
               ctx->process()->textOut()->write( *top.asString() );
            }
            else
            {
               ctx->process()->textOut()->write( "<?>" );
            }
         }
         else
         {
            // a simple item.
            String temp;
            top.describe( temp, 1, 0 );
            ctx->process()->textOut()->write( temp );
         }
      }

      // in any case, the data is gone.
      ctx->popData();
   }

   // was this the last item?
   if( pl.size() <= (size_t) seqId )
   {
      if( self->m_bAddNL )
      {
         ctx->process()->textOut()->write("\n");
      }
      ctx->process()->textOut()->flush();
      ctx->pushData(Item());
      ctx->popCode();
   }
   else
   {
      // produce the next one.
      ctx->currentCode().m_seqId = seqId + 1;
      ctx->pushCode( pl[seqId] );
   }
   self->m_mtx.unlock();
}

}

/* end of stmtfastprint.cpp */
