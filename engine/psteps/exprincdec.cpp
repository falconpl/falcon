/*
   FALCON - The Falcon Programming Language.
   FILE: exprincdec.cpp

   Standard misc PSteps commonly used in the virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 31 Jul 2011 00:39:23 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC 
#define SRC "engine/psteps/exprincdec.cpp"

#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/class.h>
#include <falcon/stdsteps.h>

#include <falcon/psteps/exprincdec.h>
#include <falcon/errors/operanderror.h>

namespace Falcon
{

class ExprPreInc::ops
{
public:
   inline bool isPre() const { return true; }
   inline static int64 operate( int64 a ) { return a + 1; }
   inline static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_inc(ctx, inst); }
   inline static numeric operaten( numeric a ) { return a + 1.0; }
   inline static const char* id() { return "++"; }
   inline static void postAssign( VMContext* ) {}
};


class ExprPreDec::ops
{
public:
   inline bool isPre() const { return true; }
   inline static int64 operate( int64 a ) { return a - 1; }
   inline static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_dec(ctx, inst); }
   inline static numeric operaten( numeric a ) { return a - 1.0; }
   inline static const char* id() { return "--"; }
   inline static void postAssign( VMContext* ) {}
};


class ExprPostInc::ops
{
public:
   inline bool isPre() const { return false; }
   inline static int64 operate( int64 a ) { return a + 1; }
   inline static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_incpost(ctx, inst); }
   inline static numeric operaten( numeric a ) { return a + 1.0; }
   inline static const char* id() { return "<post>++"; }
   inline static void postAssign(VMContext* ctx) { ctx->popData(); } 
};


class ExprPostDec::ops
{
public:
   inline bool isPre() const { return false; }
   inline static int64 operate( int64 a ) { return a - 1; }
   inline static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_decpost(ctx, inst); }
   inline static numeric operaten( numeric a ) { return a - 1.0; }
   inline static const char* id() { return "<post>--"; }
   inline static void postAssign(VMContext* ctx) { ctx->popData(); } 
};

//==============================================================
// Genetic operands


// Inline class to simplify
template <class _cpr >
bool generic_simplify( Item& value, Expression* first )
{
   if( first->simplify( value ) )
   {
      switch( value.type() )
      {
      case FLC_ITEM_INT: value.setInteger( _cpr::operate(value.asInteger()) ); return true;
      case FLC_ITEM_NUM: value.setNumeric( _cpr::operaten(value.asNumeric()) ); return true;
      }
   }

   return false;
}


// Inline class to apply
template <class _cpr >
void generic_apply_( const PStep* ps, VMContext* ctx )
{ 
   const UnaryExpression* self = static_cast<const UnaryExpression*>(ps);
#ifndef NDEBUG
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );
#endif

   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
      // Phase 0 -- generate the item.
   case 0:
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->first(), cf ) )
      {
         return;
      }
      // fallthrough
   
      // Phase 1 -- operate
   case 1:
      cf.m_seqId = 2;
      {         
         // No need to copy the second, we're not packing the stack now.
         Item *op;
         ctx->operands( op );

         // we dereference also op1 to help copy-on-write decisions from overrides
         if( op->isReference() )
         {
            *op = *op->dereference();
         }

         switch( op->type() )
         {
            case FLC_ITEM_INT: op->setInteger( _cpr::operate(op->asInteger()) ); break;
            case FLC_ITEM_NUM: op->setNumeric( _cpr::operaten(op->asNumeric()) ); break;
            case FLC_ITEM_USER:
               _cpr::operate( ctx, op->asClass(), op->asInst() );
               // went deep?
               if( &cf != &ctx->currentCode() )
               {
                  // s_nextApply will be called
                  return;
               }
               break;

            default:
            // no need to throw, we're going to get back in the VM.
            throw
               new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra(_cpr::id()) );
         }
      }
      
      // fallthrough
   
      // Phase 2 -- assigning the topmost value back.
   case 2:
      cf.m_seqId = 3;
      // now assign the topmost item in the stack to the lvalue of self.
      PStep* lvalue = self->lvalueStep();
      if( lvalue != 0 )
      {
         if( ctx->stepInYield( lvalue, cf ) )
         {
            return;
         }
      }
      
   }
   
   // eventually pop the stack -- always the code ... 
   ctx->popCode();
   // ... and possibly the parameter on the stack 
   _cpr::postAssign( ctx );
}


//=========================================================
// Implementation -- preinc
//

bool ExprPreInc::simplify( Item& value ) const
{
   return generic_simplify<ExprPreInc::ops>( value, m_first );
}


void ExprPreInc::apply_( const PStep* ps, VMContext* ctx )
{  
   generic_apply_<ExprPreInc::ops>(ps, ctx);
}


void ExprPreInc::describeTo( String& str ) const
{
   str = "++";
   str += m_first->describe();
}

//=========================================================
// Implementation -- postinc
//

bool ExprPostInc::simplify( Item& value ) const
{
   return generic_simplify<ExprPostInc::ops>( value, m_first );
}


void ExprPostInc::apply_( const PStep* ps, VMContext* ctx )
{  
   generic_apply_<ExprPostInc::ops>(ps, ctx);
}


void ExprPostInc::describeTo( String& str ) const
{
   str = m_first->describe();
   str += "++";
}

//=========================================================
// Implementation -- predec
//

bool ExprPreDec::simplify( Item& value ) const
{
   return generic_simplify<ExprPreDec::ops>( value, m_first );
}


void ExprPreDec::apply_( const PStep* ps, VMContext* ctx )
{  
   generic_apply_<ExprPreDec::ops>(ps, ctx);
}


void ExprPreDec::describeTo( String& str ) const
{
   str = "--";
   str += m_first->describe();
}

//=========================================================
// Implementation -- postdec
//

bool ExprPostDec::simplify( Item& value ) const
{
   return generic_simplify<ExprPostDec::ops>( value, m_first );
}


void ExprPostDec::apply_( const PStep* ps, VMContext* ctx )
{  
   generic_apply_<ExprPostDec::ops>(ps, ctx);
}


void ExprPostDec::describeTo( String& str ) const
{
   str = m_first->describe();
   str += "--";
}

}

/* end of exprincdec.cpp */
