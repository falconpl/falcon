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

#include <deque>

namespace Falcon
{

class StmtFastPrint::Private
{
public:
   typedef std::deque<Expression*> ExprList;   
   ExprList m_exprs;
   
   Private() {}
   ~Private() 
   {
      ExprList::iterator iter = m_exprs.begin();
      while( iter != m_exprs.end() )
      {
         delete *iter;
         ++iter;
      }
   }
};


StmtFastPrint::StmtFastPrint( bool bAddNL, int line, int chr ):
   Statement( line, chr ),
   _p( new Private ),   
   m_bAddNL( bAddNL )
{
   static Class* mycls = &Engine::instance()->synclasses()->m_stmt_fastprint;
   m_class = mycls;
   
   apply = apply_;
}


StmtFastPrint::~StmtFastPrint()
{
   delete _p;
}
   

void StmtFastPrint::add( Expression* expr )
{
   _p->m_exprs.push_back( expr );
}


Expression* StmtFastPrint::at( int n ) const
{
   return _p->m_exprs[ n ];
}


length_t StmtFastPrint::size() const
{
   return _p->m_exprs.size();
}


void StmtFastPrint::describeTo( String& str, int depth ) const
{
   str = String( " " ).replicate( depth * depthIndent ) + 
      (m_bAddNL ? "> " : ">> ");
   
   Private::ExprList::iterator iter = _p->m_exprs.begin();
   while( iter != _p->m_exprs.end() )
   {
      str += (*iter)->describe( depth + 1 );
      ++iter;
      if( iter != _p->m_exprs.end() )
      {
         str += ", ";
      }
   }
}


void StmtFastPrint::oneLinerTo( String& str ) const
{
   str = (m_bAddNL ? "> " : ">> ");
   
   Private::ExprList::iterator iter = _p->m_exprs.begin();
   while( iter != _p->m_exprs.end() )
   {
      str += (*iter)->oneLiner();
      ++iter;
      if( iter != _p->m_exprs.end() )
      {
         str += ", ";
      }
   }
}


void StmtFastPrint::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtFastPrint* self = static_cast< const StmtFastPrint*>( ps );   
   Private::ExprList& pl = self->_p->m_exprs;
   CodeFrame& cframe = ctx->currentCode();
   int seqId = cframe.m_seqId;
   
   // can we print?
   if( seqId > 0 )
   {
      register Item& top = ctx->topData();
      
      if( top.isReference() ) 
      {
         top = *top.asReference();
      }
      
      if( top.isString() )
      {
         ctx->vm()->textOut()->write( *top.asString() );
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
               ctx->vm()->textOut()->write( *top.asString() );
            }
            else
            {
               ctx->vm()->textOut()->write( "<?>" );
            }
         }
         else
         {
            // a simple item.
            String temp;
            top.describe( temp, 1, 0 );
            ctx->vm()->textOut()->write( temp );
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
         ctx->vm()->textOut()->write("\n");
      }
      ctx->popCode();
   }
   else
   {
      // produce the next one.
      cframe.m_seqId = seqId + 1;
      ctx->pushCode( pl[seqId] );
   }
}

}

/* end of stmtfastprint.cpp */
