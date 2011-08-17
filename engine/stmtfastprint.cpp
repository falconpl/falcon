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

#include <falcon/stmtfastprint.h>
#include <falcon/expression.h>
#include <falcon/pcode.h>
#include <falcon/vm.h>
#include <falcon/textwriter.h>

#include <deque>

namespace Falcon
{

class StmtFastPrint::Private
{
public:
   typedef std::deque<Expression*> ExprList;   
   ExprList m_exprs;
   
   typedef std::deque<PCode*> PCodeList;   
   PCodeList m_pcodes;
   
   Private() {}
   ~Private() 
   {
      ExprList::iterator iter = m_exprs.begin();
      while( iter != m_exprs.end() )
      {
         delete *iter;
         ++iter;
      }
      
      PCodeList::iterator piter = m_pcodes.begin();
      while( piter != m_pcodes.end() )
      {
         delete *piter;
         ++piter;
      }
   }
};


StmtFastPrint::StmtFastPrint( bool bAddNL ):
   Statement( e_stmt_fastprint ),
   _p( new Private ),   
   m_bAddNL( bAddNL )
{
   apply = apply_;
   m_step0 = this;
}


StmtFastPrint::~StmtFastPrint()
{
   delete _p;
}
   

void StmtFastPrint::add( Expression* expr )
{
   PCode* pc = new PCode;
   expr->precompile( pc );
   _p->m_exprs.push_back( expr );
   _p->m_pcodes.push_back( pc );
}


Expression* StmtFastPrint::at( int n ) const
{
   return _p->m_exprs[ n ];
}


length_t StmtFastPrint::size() const
{
   return _p->m_exprs.size();
}
   

void StmtFastPrint::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtFastPrint* self = static_cast< const StmtFastPrint*>( ps );   
   register Private::PCodeList& pl = self->_p->m_pcodes;
   register CodeFrame& cframe = ctx->currentCode();
   register int seqId = cframe.m_seqId;
   
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
