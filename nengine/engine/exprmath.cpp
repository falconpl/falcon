/*
   FALCON - The Falcon Programming Language.
   FILE: exprmath.cpp

   Expression elements -- Math basic ops.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 23:35:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/exprmath.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/vm.h>
#include <falcon/operanderror.h>
#include <falcon/codeerror.h>

#include <math.h>

namespace Falcon {

class ExprPlus::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a + b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_add(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a + b; }
   static bool zeroCheck( const Item& ) { return false; }
};

class ExprMinus::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a - b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_sub(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a - b; }
   static bool zeroCheck( const Item& ) { return false; }
};

class ExprTimes::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a * b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_mul(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a * b; }
   static bool zeroCheck( const Item& ) { return false; }
};

class ExprDiv::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a / b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_div(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a / b; }
   static bool zeroCheck( const Item& n ) { return n.isOrdinal() && n.forceNumeric() == 0.0; }
};

class ExprMod::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a % b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_mod(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return ((int64)a) % ((int64)b); }
   static bool zeroCheck( const Item& n ) { return n.isOrdinal() && n.forceNumeric() == 0.0; }
};

class ExprPow::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return (int64)pow(a,(numeric)b); }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_pow(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return pow(a,b); }
   static bool zeroCheck( const Item& ) { return false; }
};


// Inline class to simplify
template <class __CPR >
bool generic_simplify( Item& value, Expression* m_first, Expression* m_second )
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_second->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setInteger( __CPR::operate( d1.asInteger(), d2.asInteger() ) );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setNumeric( __CPR::operaten( d1.asInteger(), d2.asNumeric() ) );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setNumeric( __CPR::operaten( d1.asNumeric(), d2.asInteger() ) ) ;
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setNumeric( __CPR::operaten( d1.asNumeric(), d2.asNumeric() ) );
         return true;
      }
   }

   return false;
}


// Inline class to apply
template <class __CPR >
void generic_apply_( const PStep* ps, VMContext* ctx )
{
   TRACE2( "Apply \"%s\"", ((ExprMath*)ps)->describe().c_ize() );

   // No need to copy the second, we're not packing the stack now.
   Item *op1, *op2;
   ctx->operands( op1, op2 );

   if ( __CPR::zeroCheck(*op2) )
   {
      throw new CodeError( ErrorParam(e_div_by_zero, __LINE__).origin(ErrorParam::e_orig_vm) );
   }

   switch ( op1->type() << 8 | op2->type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      op1->setInteger( __CPR::operate(op1->asInteger(), op2->asInteger()) );
      ctx->popData();
      break;

   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      op1->setNumeric( __CPR::operaten(op1->asInteger(), op2->asNumeric()) );
      ctx->popData();
      break;
   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      op1->setNumeric( __CPR::operaten(op1->asNumeric(), op2->asInteger()) );
      ctx->popData();
      break;
   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      op1->setNumeric( __CPR::operaten(op1->asNumeric(), op2->asNumeric()) );
      ctx->popData();
      break;

   case FLC_ITEM_USER << 8 | FLC_ITEM_NIL:
   case FLC_ITEM_USER << 8 | FLC_ITEM_BOOL:
   case FLC_ITEM_USER << 8 | FLC_ITEM_INT:
   case FLC_ITEM_USER << 8 | FLC_ITEM_NUM:
   case FLC_ITEM_USER << 8 | FLC_ITEM_METHOD:
   case FLC_ITEM_USER << 8 | FLC_ITEM_FUNC:
   case FLC_ITEM_USER << 8 | FLC_ITEM_BASEMETHOD:
   case FLC_ITEM_USER << 8 | FLC_ITEM_USER:
      __CPR::operate( ctx, op1->asClass(), op1->asInst() );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      throw
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra(((ExprMath*)ps)->name()) );
   }
}

template
void generic_apply_<ExprPlus::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprMinus::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprTimes::ops>( const PStep* ps, VMContext* ctx);

template
void generic_apply_<ExprDiv::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprMod::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprPow::ops>( const PStep* ps, VMContext* ctx );

//==========================================================


ExprMath::ExprMath( Expression* op1, Expression* op2, Expression::operator_t t, const String& name ):
   BinaryExpression( t, op1, op2 ),
   m_name(name)
{}

ExprMath::ExprMath( const ExprMath& other ):
   BinaryExpression( other ),
   m_name( other.m_name )
{}

ExprMath::~ExprMath()
{}


void ExprMath::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + m_name + m_second->describe() + ")";
}

//========================================================
// EXPR Plus
//

ExprPlus::ExprPlus( Expression* op1, Expression* op2 ):
   ExprMath( op1, op2, t_plus, "+" )
{
   apply = &generic_apply_<ops>;
}

ExprPlus::~ExprPlus()
{}

bool ExprPlus::simplify( Item& value ) const
{
   return generic_simplify<ops>( value, m_first, m_second );
}


//========================================================
// EXPR Minus
//
ExprMinus::ExprMinus( Expression* op1, Expression* op2 ):
   ExprMath( op1, op2, t_le, "-" )
{
   apply = &generic_apply_<ops>;
}


ExprMinus::~ExprMinus()
{}

bool ExprMinus::simplify( Item& value ) const
{
   return generic_simplify<ops>( value, m_first, m_second );
}

//========================================================
// EXPR Times
//
ExprTimes::ExprTimes( Expression* op1, Expression* op2 ):
   ExprMath( op1, op2, t_gt, "*" )
{
   apply = &generic_apply_<ops>;
}

ExprTimes::~ExprTimes()
{}

bool ExprTimes::simplify( Item& value ) const
{
   return generic_simplify<ops>( value, m_first, m_second );
}

//========================================================
// EXPR Div
//
ExprDiv::ExprDiv( Expression* op1, Expression* op2 ):
   ExprMath( op1, op2, t_ge, "/" )
{
   apply = &generic_apply_<ops>;
}

ExprDiv::~ExprDiv()
{}

bool ExprDiv::simplify( Item& value ) const
{
   return generic_simplify<ops>( value, m_first, m_second );
}

//========================================================
// EXPR Mod
//
ExprMod::ExprMod( Expression* op1, Expression* op2 ):
   ExprMath( op1, op2, t_eq, "%" )
{
   apply = &generic_apply_<ops>;
}

ExprMod::~ExprMod()
{}

bool ExprMod::simplify( Item& value ) const
{
   return generic_simplify<ops>( value, m_first, m_second );
}


//========================================================
// EXPR Pow
//
ExprPow::ExprPow( Expression* op1, Expression* op2 ):
   ExprMath( op1, op2, t_neq, "**" )
{
   apply = &generic_apply_<ops>;
}

ExprPow::~ExprPow()
{}


bool ExprPow::simplify( Item& value ) const
{
   return generic_simplify<ops>( value, m_first, m_second );
}


}

/* end of exprmath.cpp */

