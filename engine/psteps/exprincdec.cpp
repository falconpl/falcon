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
#include <falcon/pcode.h>

#include <falcon/psteps/exprincdec.h>
#include <falcon/errors/operanderror.h>

namespace Falcon
{

class ExprPreInc::ops
{
public:
   static int64 operate( int64 a ) { return a + 1; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_inc(ctx, inst); }
   static numeric operaten( numeric a ) { return a + 1.0; }
   static const char* id() { return "++"; }
};


class ExprPreDec::ops
{
public:
   static int64 operate( int64 a ) { return a - 1; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_dec(ctx, inst); }
   static numeric operaten( numeric a ) { return a - 1.0; }
   static const char* id() { return "--"; }
};


class ExprPostInc::ops
{
public:
   static int64 operate( int64 a ) { return a + 1; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_incpost(ctx, inst); }
   static numeric operaten( numeric a ) { return a + 1.0; }
   static const char* id() { return "<post>++"; }
};


class ExprPostDec::ops
{
public:
   static int64 operate( int64 a ) { return a - 1; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_decpost(ctx, inst); }
   static numeric operaten( numeric a ) { return a - 1.0; }
   static const char* id() { return "<post>--"; }
};



//==============================================================
// Genetic operands


// Inline class to simplify
template <class __CPR >
bool generic_simplify( Item& value, Expression* first )
{
   if( first->simplify( value ) )
   {
      switch( value.type() )
      {
      case FLC_ITEM_INT: value.setInteger( __CPR::operate(value.asInteger()) ); return true;
      case FLC_ITEM_NUM: value.setNumeric( __CPR::operaten(value.asNumeric()) ); return true;
      }
   }

   return false;
}


// Inline class to apply
template <class __CPR >
void generic_apply_( const PStep* DEBUG_ONLY(ps), VMContext* ctx )
{
#ifndef NDEBUG
   UnaryExpression* un = (UnaryExpression*)ps;
   TRACE2( "Apply \"%s\"", un->describe().c_ize() );
#endif
   
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
      case FLC_ITEM_INT: op->setInteger(  __CPR::operate(op->asInteger()) ); break;
      case FLC_ITEM_NUM: op->setNumeric(  __CPR::operaten(op->asNumeric()) ); break;
      case FLC_ITEM_USER:
         __CPR::operate( ctx, op->asClass(), op->asInst() );
         break;

      default:
      // no need to throw, we're going to get back in the VM.
      throw
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra(__CPR::id()) );
   }
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


void ExprPreInc::precompile( PCode* pcode ) const
{
   TRACE2( "Precompile \"%s\"", describe().c_ize() );   
   m_first->precompileAutoLvalue( pcode, this, false, false );
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


void ExprPostInc::precompile( PCode* pcode ) const
{
   static StdSteps* stdSteps = Engine::instance()->stdSteps();
   
   TRACE2( "Precompile \"%s\"", describe().c_ize() );
   pcode->pushStep( &stdSteps->m_addSpace_ );
   m_first->precompileAutoLvalue( pcode, this, false, true );
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

void ExprPreDec::precompile( PCode* pcode ) const
{
   TRACE2( "Precompile \"%s\"", describe().c_ize() );
   m_first->precompileAutoLvalue( pcode, this, false, false );
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


void ExprPostDec::precompile( PCode* pcode ) const
{
   static StdSteps* stdSteps = Engine::instance()->stdSteps();
   TRACE2( "Precompile \"%s\"", describe().c_ize() );
   
   pcode->pushStep( &stdSteps->m_addSpace_ );
   m_first->precompileAutoLvalue( pcode, this, false, true );
}

void ExprPostDec::describeTo( String& str ) const
{
   str = m_first->describe();
   str += "--";
}

}

/* end of exprincdec.cpp */
