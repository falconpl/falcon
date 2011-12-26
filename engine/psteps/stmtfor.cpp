/*
   FALCON - The Falcon Programming Language.
   FILE: stmtfor.cpp

   Syntactic tree item definitions -- Autoexpression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Aug 2011 17:28:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/stmtfor.cpp"

#include <falcon/trace.h>
#include <falcon/fassert.h>
#include <falcon/expression.h>
#include <falcon/stdsteps.h>
#include <falcon/syntree.h>
#include <falcon/symbol.h>
#include <falcon/itemarray.h>

#include <falcon/errors/accesserror.h>
#include <falcon/errors/codeerror.h>

#include <falcon/psteps/stmtfor.h>

#include <vector>

namespace Falcon 
{


StmtForBase::~StmtForBase()
{
   delete m_body;
   delete m_forFirst;
   delete m_forMiddle;
   delete m_forLast;
}

void StmtForBase::PStepCleanup::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->popData(ctx->currentCode().m_seqId);
}

void StmtForBase::describeTo( String& tgt, int depth ) const
{   
   String prefix = String(" ").replicate( depth * depthIndent );   
   String prefix1 = String(" ").replicate( (depth+1) * depthIndent );
   tgt =  + oneLiner();
   tgt += "\n";
   
   if( m_body != 0 )
   {
      String temp;
      m_body->describeTo( temp, depth + 1);
      tgt += prefix1 + temp + "\n";
   }
   
   if( m_forFirst != 0 )
   {
      String temp;
      m_forFirst->describeTo( temp, depth + 1 );
      tgt += prefix + "forfirst\n" + prefix1 +  temp + "\n" + prefix + "end\n";
   }
   
   if( m_forMiddle != 0 )
   {
      String temp;
      m_forMiddle->describeTo( temp, depth + 1 );
      tgt+= prefix + "formiddle\n" + prefix1 +  temp + "\n" + prefix + "end\n";
   }
   
   if( m_forLast != 0 )
   {
      String temp;
      m_forLast->describeTo( temp, depth + 1 );
      tgt += prefix + "forlast\n" + prefix1 +  temp + "\n" + prefix + "end\n";
   }
   
   tgt += prefix + "end";
}


//=================================================================
// For - in
//

class StmtForIn::Private
{
public:
   typedef std::vector<Symbol*> SymVector;
   SymVector m_params;
   
   Private() {}
   ~Private() {}
};


StmtForIn::StmtForIn( Expression* gen, int32 line, int32 chr):
   StmtForBase( Statement::e_stmt_for_in, line, chr ),
   _p( new Private ),
   m_expr(gen),
   m_stepBegin( this ),
   m_stepFirst( this ),
   m_stepNext( this ),
   m_stepGetNext( this )
{
   apply = apply_; 
   
   //NOTE: This pstep is NOT a loopbase; it just sets up the loop.
   // a break here must be intercepted by outer loops until our loop is setup.
   
}

StmtForIn::~StmtForIn()
{
   delete _p;
}


void StmtForIn::oneLinerTo( String& tgt ) const
{
   tgt += "for ";
      
   String syms;
   Private::SymVector::const_iterator iter = _p->m_params.begin();
   while( iter != _p->m_params.end() )
   {
      if( syms.size() != 0 )
      {
         syms += ", ";
      }
      syms += (*iter)->name();
      ++iter;
   }
   
   tgt += syms;
   tgt += " in ";
   fassert( m_expr != 0 );
   if( m_expr != 0 ) tgt += m_expr->describe();
}


void StmtForIn::addParameter( Symbol* sym )
{
   _p->m_params.push_back( sym );
}


length_t StmtForIn::paramCount() const
{
   return (length_t) _p->m_params.size();
}

Symbol* StmtForIn::param( length_t p ) const
{
   return _p->m_params[p];
}


void StmtForIn::expandItem( Item& itm, VMContext* ctx ) const
{
   if( _p->m_params.size() == 1 )
   {
      _p->m_params[0]->value( ctx )->assign( itm );
   }
   else
   {
      Item* dr = itm.dereference();   
      if( dr->isArray() )
      {
         ItemArray* ar = dr->asArray();
         if( ar->length() == _p->m_params.size() )
         {
            for( length_t i = 0; i < ar->length(); ++i )
            {
               _p->m_params[i]->value( ctx )->assign( (*ar)[i] );               
            }
            return;
         }
         
      }
      
      // failed...
      throw new AccessError( ErrorParam( e_unpack_size, __LINE__, SRC )
         .origin(ErrorParam::e_orig_vm)
         .extra( "for/in") 
         );
   }
}

void StmtForIn::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForIn* self = static_cast<const StmtForIn*>(ps);
   ctx->resetCode( &self->m_stepBegin );
   ctx->stepIn( self->m_expr );
}

void StmtForIn::PStepBegin::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForIn* self = static_cast<const StmtForIn::PStepBegin*>(ps)->m_owner;
   
   // we have the evaluated expression on top of the stack -- make it to next.
   Class* cls;
   void* dt;
   if( ctx->topData().asClassInst( cls, dt )  )
   {       
      // Prepare to get the iterator item...
      ctx->currentCode().m_step = &self->m_stepCleanup;
      ctx->currentCode().m_seqId = 2;
      ctx->pushCode( &self->m_stepFirst );      
      ctx->pushCode( &self->m_stepGetNext );
      
      // and create an iterator.
      //ctx->addLocals(2);      
      cls->op_iter( ctx, dt );       
   }
   else if( ctx->topData().isNil() )
   {
      // Nil is defined to cleanly exit the loop.
      ctx->popCode();
   }
   else
   {
      throw new CodeError( 
         ErrorParam( e_not_iterable, __LINE__, SRC )
         .origin( ErrorParam::e_orig_vm )
         .extra( "for/in" ) );
   }    
}


void StmtForIn::PStepGetNext::apply_( const PStep*, VMContext* ctx )
{
    fassert( ctx->opcodeParam(1).isUser() );
       
    // we're never needed anymore
    ctx->popCode();
    
    Class* cls = 0;
    void* dt = 0;
    // here we have seq, iter, <space>...
    ctx->opcodeParam(1).asClassInst( cls, dt );
    // ... pass them to next.
    cls->op_next( ctx, dt );
}


void StmtForIn::PStepFirst::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForIn* self = static_cast<const StmtForIn::PStepFirst*>(ps)->m_owner;

   // we have here seq, iter, item
   Item& topData = ctx->topData();
   if( topData.isBreak() )
   {
      ctx->popCode();
      return;
   }

   // prepare the loop variabiles.
   self->expandItem( ctx->topData(), ctx );
   
   if( topData.isLast() )
   {
      topData.flagsOff( Item::flagLast );
      ctx->popCode(); // we won't be called again.
      if( self->m_forLast != 0 )
      {
         ctx->pushCode( self->m_forLast );
      }
   }
   else
   {
      ctx->currentCode().m_step = &self->m_stepNext;
      ctx->pushCode( &self->m_stepGetNext );
      
      if ( self->m_forMiddle != 0 )
      {
         ctx->pushCode( self->m_forMiddle );
      }
   }
   
   if ( self->m_body != 0 )
   {
      ctx->pushCode( self->m_body );
   }

   if ( self->m_forFirst != 0 )
   {
      ctx->pushCode( self->m_forFirst );
   }
   
   // in any case, the extra item can be removed.
   ctx->popData();
}


void StmtForIn::PStepNext::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForIn* self = static_cast<const StmtForIn::PStepNext*>(ps)->m_owner;

   // we have here seq, iter, item
   Item& topData = ctx->topData();
   if( topData.isBreak() )
   {
      ctx->popCode();
      return;
   }
   
   // prepare the loop variabiles.
   self->expandItem( ctx->topData(), ctx );

   if( topData.isLast() )
   {
      topData.flagsOff( Item::flagLast );
      ctx->popCode(); // we won't be called again.
       
      if( self->m_forLast != 0 )
      {
         ctx->pushCode( self->m_forLast );
      }
   }
   else
   {      
      ctx->pushCode( &self->m_stepGetNext );
      if ( self->m_forMiddle != 0 )
      {
         ctx->pushCode( self->m_forMiddle );
      }
   }
      
   if ( self->m_body != 0 )
   {
      ctx->pushCode( self->m_body );
   }
   // in any case, the extra item can be removed.
   ctx->popData();
}


//===============================================
// For - to
//

StmtForTo::StmtForTo( Symbol* tgt, Expression* start, Expression* end, Expression* step, int32 line, int32 chr ):
   StmtForBase( Statement::e_stmt_for_to, line, chr ),
   m_target( tgt ),
   m_start(start),
   m_end(end),  
   m_step(step),
   m_stepNext(this)
{
   apply = apply_;     
}


StmtForTo::~StmtForTo() 
{
   delete m_start;
   delete m_end;
   delete m_step;  
}
      

void StmtForTo::startExpr( Expression* s )
{
   delete m_start;
   m_start = s;
}


void StmtForTo::endExpr( Expression* s )
{
   delete m_end;
   m_end = s;
}
   
void StmtForTo::stepExpr( Expression* s )
{
   delete m_step;
   m_step = s;
}

   
void StmtForTo::oneLinerTo( String& tgt ) const
{  
   tgt += "for " + m_target->name() + " = " ;

   tgt += m_start->describe();   
   tgt + " to ";   
   tgt += m_end->describe();
   if( m_step != 0 )
   {
      tgt += ", " + m_step->describe();
   }
}
   

void StmtForTo::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForTo* self = static_cast<const StmtForTo*>(ps);
   
   // we must at least have a start and an end
   fassert( self->m_start != 0 );
   fassert( self->m_end != 0 );
   
   // First of all, start executing the start, end and step expressions.
   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
      case 0: 
         // check the start.
         cf.m_seqId = 1;
         if( ctx->stepInYield( self->m_start, cf ) )
         {
            return;
         }
         // fallthrough
      case 1:
         cf.m_seqId = 2;
         if( ctx->stepInYield( self->m_end, cf ) )
         {
            return;
         }
         // fallthrough
      case 2:
         cf.m_seqId = 3;
         if( self->m_step != 0 )
         {
            if( ctx->stepInYield( self->m_step, cf ) )
            {
               return;
            }
         }
         else {
            ctx->pushData( (int64) 0 );
         }
   }
   
   int64 step = ctx->topData().asInteger();
   int64 end = ctx->opcodeParam(1).asInteger();
   int64 start = ctx->opcodeParam(2).asInteger();
   
   // in some cases, we don't even start the loop
   if( (end > start && step < 0) || (start > end && step > 0 ) )
   {
      ctx->popCode();
      ctx->popData(3);
      return;
   }
   
   // fix the default step
   if( step == 0 )
   {
      if ( end >= start ) 
      {
         ctx->opcodeParam(0).setInteger(1);
      }
      else
      {
         ctx->opcodeParam(0).setInteger(-1);
      }      
   }
   
   // however, we won't be called anymore.
   cf.m_step = &self->m_stepCleanup;
   cf.m_seqId = 3; // 3 items to remove at cleanup
   ctx->pushCode( &self->m_stepNext );
   
   // Prepare the start value   
   Symbol* target = self->m_target;
   Item* tgtItem = target->value( ctx );
   tgtItem->setInteger( start );   
   
   // eventually, push the first opode in top of all.
   if( self->m_forFirst != 0 )
   {
      ctx->pushCode( self->m_forFirst );
   }
}
 

void StmtForTo::PStepNext::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForTo* self = static_cast<const StmtForTo::PStepNext*>(ps)->m_owner;
   
   register int64 start = ctx->opcodeParam(2).asInteger();
   int64 end = ctx->opcodeParam(1).asInteger();
   int64 step = ctx->topData().asInteger();
   
   // the start, at minimum, will be done.
   Symbol* target = self->m_target;
   Item* tgtItem = target->value( ctx );
   tgtItem->setInteger( start );   
   
   // step cannot be 0 as it has been sanitized by our main step.
   if( (step > 0 && start >= end) || ( step < 0 && start <= end ) )
   {
      // this will be the last loop.
      ctx->popCode();
      
      if( self->m_forLast != 0 )
      {
         ctx->pushCode( self->m_forLast );
      }
   }
   else
   {
      if( self->m_forMiddle != 0 )
      {
         ctx->pushCode( self->m_forMiddle );
      }
   }
   
   if( self->m_body )
   {
      ctx->pushCode( self->m_body );
   }
   
   start += step;
   ctx->opcodeParam(2).content.data.val64 = start;
}


}

/* end of stmtfor.cpp */
