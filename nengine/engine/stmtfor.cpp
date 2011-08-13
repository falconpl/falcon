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
#define SRC "engine/stmtfor.cpp"

#include <falcon/trace.h>
#include <falcon/fassert.h>
#include <falcon/stmtfor.h>
#include <falcon/expression.h>
#include <falcon/codeerror.h>
#include <falcon/stdsteps.h>
#include <falcon/syntree.h>
#include <falcon/symbol.h>
#include <falcon/itemarray.h>
#include <falcon/accesserror.h>

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


void StmtForBase::describeTo( String& tgt ) const
{   
   oneLinerTo(tgt);
   tgt += "\n";
   
   if( m_body != 0 )
   {
      String temp;
      m_body->describeTo( temp );
      tgt+=temp;
   }
   
   if( m_forFirst != 0 )
   {
      String temp;
      m_forFirst->describeTo( temp );
      tgt += "\nforfirst\n" + temp + "end\n\n";
   }
   
   if( m_forMiddle != 0 )
   {
      String temp;
      m_forMiddle->describeTo( temp );
      tgt+= "\nformiddle\n" + temp + "end\n\n";
   }
   
   if( m_forLast != 0 )
   {
      String temp;
      m_forLast->describeTo( temp );
      tgt+= "\nforlast\n" + temp + "end\n\n";
   }
   
   tgt += "end\n\n";
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
   m_stepFirst( this ),
   m_stepNext( this ),
   m_stepGetNext( this )
{
   apply = apply_;
   m_bIsLoopBase = true;
   m_step0 = this;
   gen->precompile( &m_pcExpr );
   m_step1 = &m_pcExpr;
   
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
   
   // we have the evaluated expression on top of the stack -- make it to next.
   Class* cls;
   void* dt;
   if( ctx->topData().asClassInst( cls, dt )  )
   {       
      // Prepare to get the iterator item...
      ctx->currentCode().m_step = &self->m_stepFirst;      
      ctx->pushCode( &self->m_stepGetNext );
      
      // and create an iterator.
      ctx->addLocals(2);      
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
    fassert( ctx->opcodeParam(2).isUser() );
       
    // we're never needed anymore
    ctx->popCode();
    
    Class* cls = 0;
    void* dt = 0;
    // here we have seq, iter, <space>...
    ctx->opcodeParam(2).asClassInst( cls, dt );
    // ... pass them to next.
    cls->op_next( ctx, dt );
}


void StmtForIn::PStepFirst::apply_( const PStep* ps, VMContext* ctx )
{
   static PStep* spop3 = &Engine::instance()->stdSteps()->m_pop3;
   
   const StmtForIn* self = static_cast<const StmtForIn::PStepFirst*>(ps)->m_owner;

   // we have here seq, iter, item
   Item& topData = ctx->topData();
   if( topData.isBreak() )
   {
      ctx->popCode();
      ctx->popData(3);
      return;
   }

   // prepare the loop variabiles.
   self->expandItem( ctx->topData(), ctx );
   
   if( topData.isLast() )
   {
      topData.flagsOff( Item::flagLast );
      ctx->currentCode().m_step = spop3;
       
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
}


void StmtForIn::PStepNext::apply_( const PStep* ps, VMContext* ctx )
{
   static PStep* spop3 = &Engine::instance()->stdSteps()->m_pop3;
   
   const StmtForIn* self = static_cast<const StmtForIn::PStepNext*>(ps)->m_owner;

   // we have here seq, iter, item
   Item& topData = ctx->topData();
   if( topData.isBreak() )
   {
      ctx->popCode();
      ctx->popData(3);
      return;
   }
   
   // prepare the loop variabiles.
   self->expandItem( ctx->topData(), ctx );

   if( topData.isLast() )
   {
      topData.flagsOff( Item::flagLast );
      ctx->currentCode().m_step = spop3;
       
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
}


//===============================================
// For - to
//

StmtForTo::StmtForTo( Symbol* tgt, int64 start, int64 end, int64 step, int32 line, int32 chr ):
   StmtForBase( Statement::e_stmt_for_to, line, chr ),
   m_target( tgt ),
   m_start(0),
   m_end(0),  
   m_step(0),
   m_istart(start),
   m_iend(end),
   m_istep(step),
   m_stepNext(this),
   m_stepPushStart(this),
   m_stepPushEnd(this),
   m_stepPushStep(this)
{
   apply = apply_;
   m_bIsLoopBase = true;
   
   m_step0 = this;
   m_step1 = &m_stepPushStart;
   m_step2 = &m_stepPushEnd;
   m_step3 = &m_stepPushStep;      
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
   if( s != 0 )
   {
      s->precompile(&m_pcExprStart);
      m_step1 = &m_pcExprStart;
   }
}


void StmtForTo::endExpr( Expression* s )
{
   delete m_end;
   m_end = s;
   if( s != 0 )
   {
      s->precompile(&m_pcExprEnd);
      m_step2 = &m_pcExprEnd;
   }
}
   
void StmtForTo::stepExpr( Expression* s )
{
   delete m_step;
   m_step = s;
   if( s != 0 )
   {
      s->precompile(&m_pcExprStep);
      m_step3 = &m_pcExprStep;
   }
}

   
void StmtForTo::oneLinerTo( String& tgt ) const
{  
   tgt += "for " + m_target->name() + " = " ;

   if( m_start != 0 )
   {
      tgt += m_start->describe();
   }
   else
   {
      tgt.N( m_istart );
   }
   
   tgt + " to ";
   
   if( m_end != 0 )
   {
      tgt += m_end->describe();
   }
   else
   {
      tgt.N( m_iend );
   }
   
   if( m_step != 0 )
   {
      tgt += " step " + m_step->describe();
   }
   else
   {
      tgt.A(" step ").N( m_iend );
   }
}
   

void StmtForTo::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForTo* self = static_cast<const StmtForTo*>(ps);
   int64 start = ctx->topData().asInteger();
   int64 end = ctx->opcodeParam(1).asInteger();
   int64 step = ctx->opcodeParam(2).asInteger();
   
   // in some cases, we don't even start the loop
   if( (end > start && step < 0) || (start > end && step > 0 ) || ctx->topData().isBreak())
   {
      ctx->popCode();
      ctx->popData(3);
      return;
   }
   
   bool bLast = false;
   
   // however, we won't be called anymore.
   ctx->currentCode().m_step = &self->m_stepNext;
   
   // the start, at minimum, will be done.
   self->m_target->value( ctx )->setInteger( start );   
   if( step == 0 )
   {
      if ( end > start ) 
      {
         ctx->topData().setInteger(start+1);
      }
      else if ( end < start ) 
      {
         ctx->topData().setInteger(start-1);
      }
      else
      {
         // this will be the last loop.
         ctx->popCode();
         ctx->popData(3);
         bLast = true;
      }
   }
   else
   {
      start += step;
      ctx->topData().setInteger(start);
      
      if( (step > 0 && start >= end) || ( step < 0 && start <= end ) )
      {
         // this will be the last loop.
         ctx->popCode();
         ctx->popData(3);
         bLast = true;
      }
   }
   
   if( bLast )
   {
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
   
   if( self->m_forFirst )
   {
      ctx->pushCode( self->m_forFirst );
   }
}
 

void StmtForTo::PStepNext::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForTo* self = static_cast<const StmtForTo::PStepNext*>(ps)->m_owner;

   bool bLast = false;
   
   if( ctx->topData().isBreak() )
   {
      ctx->popCode();
      ctx->popData(3);
      return;
   }
   
   int64 start = ctx->topData().asInteger();
   int64 end = ctx->opcodeParam(1).asInteger();
   int64 step = ctx->opcodeParam(2).asInteger();
      
   // however, we won't be called anymore.
   ctx->currentCode().m_step = &self->m_stepNext;
   
   // the start, at minimum, will be done.
   self->m_target->value( ctx )->setInteger( start );   
   if( step == 0 )
   {
      if ( end > start ) 
      {
         ctx->topData().setInteger(start+1);
      }
      else if ( end < start ) 
      {
         ctx->topData().setInteger(start-1);
      }
      else
      {
         // this will be the last loop.
         ctx->popCode();
         ctx->popData(3);
         bLast = true;
      }
   }
   else
   {
      start += step;
      ctx->topData().setInteger(start);
      
      if( (step > 0 && start >= end) || ( step < 0 && start <= end ) )
      {
         // this will be the last loop.
         ctx->popCode();
         ctx->popData(3);
         bLast = true;
      }
   }
   
   if( bLast )
   {
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
}


void StmtForTo::PStepPushStart::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForTo* self = static_cast<const StmtForTo::PStepPushStart*>(ps)->m_owner;
   ctx->popCode();   
   ctx->pushData( self->m_istart );
}

void StmtForTo::PStepPushEnd::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForTo* self = static_cast<const StmtForTo::PStepPushEnd*>(ps)->m_owner;
   ctx->popCode();   
   ctx->pushData( self->m_iend );
}

void StmtForTo::PStepPushStep::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForTo* self = static_cast<const StmtForTo::PStepPushStep*>(ps)->m_owner;
   ctx->popCode();   
   ctx->pushData( self->m_istep );
}


}

/* end of stmtfor.cpp */
