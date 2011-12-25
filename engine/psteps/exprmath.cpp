/*
   FALCON - The Falcon Programming Language.
   FILE: exprmath.cpp

   Expression elements -- Math basic ops.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 23:35:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.(1)
*/

#undef SRC
#define SRC "engine/psteps/exprmath.cpp"

#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/vm.h>
#include <falcon/errors/operanderror.h>
#include <falcon/errors/codeerror.h>
#include <falcon/errors/matherror.h>

#include <falcon/psteps/exprmath.h>

#include <math.h>

namespace Falcon {

class ExprPlus::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a + b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_add(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a + b; }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item& , Item&  ) {}   
};

class ExprMinus::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a - b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_sub(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a - b; }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item&  ) {}   
};

class ExprTimes::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a * b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_mul(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a * b; }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& ) {}   
};

class ExprDiv::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a / b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_div(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a / b; }
   static bool zeroCheck( const Item& n ) { return n.isOrdinal() && n.forceInteger() == 0; }
   static void swapper( Item&, Item& op1) { 
      if( op1.isInteger() ) {op1.setNumeric( op1.asInteger());} 
   }   
};

class ExprMod::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a % b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_mod(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return ((int64)a) % ((int64)b); }
   static bool zeroCheck( const Item& n ) { return n.isOrdinal() && n.forceInteger() == 0; }
   static void swapper( Item&, Item& ) {}   
};

class ExprPow::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return (int64)pow(a,(numeric)b); }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_pow(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return pow(a,b); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& op1) { 
      if( op1.isInteger() ) {op1.setNumeric( op1.asInteger());} 
   }   
};


class ExprRShift::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return ((uint64)a) >> ((uint64)b); }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_shr(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return ((uint64)a) >> ((uint64)b); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item&) {}
};

class ExprLShift::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return (int64) ((uint64)a) << ((uint64)b); }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_shl(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return ((uint64)a) << ((uint64)b); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& ) {} 
};


class ExprBAND::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return ((uint64)a) & ((uint64)b);; }
   static void operate( VMContext*, Class*, void* ) { 
      throw new OperandError( ErrorParam( e_invop, __LINE__, SRC )
         .extra("^&")
         .origin( ErrorParam::e_orig_vm )); 
   }
   static numeric operaten( numeric a, numeric b ) { return ((uint64)a) & ((uint64)b); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& ) {} 
};

class ExprBOR::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return ((uint64)a) | ((uint64)b);; }
   static void operate( VMContext*, Class*, void* ) { 
      throw new OperandError( ErrorParam( e_invop, __LINE__, SRC )
         .extra("^|")
         .origin( ErrorParam::e_orig_vm )); 
   }
   static numeric operaten( numeric a, numeric b ) { return ((uint64)a) | ((uint64)b); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& ) {} 
};

class ExprBXOR::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return ((uint64)a) ^ ((uint64)b); }
   static void operate( VMContext*, Class*, void* ) { 
      throw new OperandError( ErrorParam( e_invop, __LINE__, SRC )
         .extra("^^")
         .origin( ErrorParam::e_orig_vm )); 
   }
   static numeric operaten( numeric a, numeric b ) { return ((uint64)a) ^ ((uint64)b); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& ) {} 
};

//==================================================
// Autoexprs
//

class ExprAutoPlus::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a + b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_aadd(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a + b; }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item& op1, Item& op2 ) { op1.swap(op2); }   
};

class ExprAutoMinus::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a - b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_asub(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a - b; }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item& op1, Item& op2 ) { op1.swap(op2); }   
};

class ExprAutoTimes::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a * b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_amul(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a * b; }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item& op1, Item& op2 ) { op1.swap(op2); }   
};

class ExprAutoDiv::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a / b; }   
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_adiv(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a / b; }
   static bool zeroCheck( const Item& n ) { return n.isOrdinal() && n.forceInteger() == 0; }
   static void swapper( Item& op1, Item& op2 ) { 
      op1.swap(op2); 
      if( op2.isInteger() ) {op2.setNumeric( op2.asInteger());} 
   }   
};

class ExprAutoMod::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a % b; }   
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_amod(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return ((int64)a) % ((int64)b); }
   static bool zeroCheck( const Item& n ) { return n.isOrdinal() && n.forceInteger() == 0; }
   static void swapper( Item& op1, Item& op2 ) { op1.swap(op2); }   
};

class ExprAutoPow::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return (int64)pow(a,(numeric)b); }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_apow(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return pow(a,b); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item& op1, Item& op2 ) {
      op1.swap(op2);
      if( op2.isInteger() ) {op2.setNumeric( op2.asInteger());}
   }
};

class ExprAutoRShift::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return ((uint64)a) >> ((uint64)b);; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_ashr(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return ((uint64)a) >> ((uint64)b); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item& op1, Item& op2 ) {op1.swap(op2);}
};

class ExprAutoLShift::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return ((uint64)a) << ((uint64)b);; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_ashl(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return ((uint64)a) << ((uint64)b); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item& op1, Item& op2 ) { op1.swap(op2); }   
};



// Inline class to simplify
template <class _cpr >
bool generic_simplify( Item& value, Expression* m_first, Expression* m_second )
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_second->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setInteger( _cpr::operate( d1.asInteger(), d2.asInteger() ) );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setNumeric( _cpr::operaten( (numeric) d1.asInteger(), d2.asNumeric() ) );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setNumeric( _cpr::operaten( d1.asNumeric(),(numeric) d2.asInteger() ) ) ;
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setNumeric( _cpr::operaten( d1.asNumeric(), d2.asNumeric() ) );
         return true;
      }
   }

   return false;
}


// Inline class to apply
template <class _cpr >
void generic_apply_( const PStep* ps, VMContext* ctx )
{  
   const ExprMath* self = static_cast<const ExprMath*>(ps);
   
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
      
      // Phase 1 -- generate the other item.
   case 1:
      cf.m_seqId = 2;
      if( ctx->stepInYield( self->second(), cf ) )
      {
         return;
      }
      // fallthrough
   
      // Phase 2 -- operate
   case 2:
      cf.m_seqId = 3;
      {         
         // No need to copy the second, we're not packing the stack now.
         register Item *op1 = &ctx->opcodeParam(1); 
         register Item *op2 = &ctx->opcodeParam(0);
         _cpr::swapper( *op1, *op2 );   

         if ( _cpr::zeroCheck(*op2) )
         {
            throw new MathError( ErrorParam(e_div_by_zero, __LINE__, SRC )
               .origin(ErrorParam::e_orig_vm) );
         }

         // we dereference also op1 to help copy-on-write decisions from overrides
         if( op1->isReference() )
         {
            *op1 = *op1->dereference();
         }

         if( op2->isReference() )
         {
            *op2 = *op2->dereference();
         }

         switch ( op1->type() << 8 | op2->type() )
         {
         case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
            op1->content.data.val64 = _cpr::operate(op1->asInteger(), op2->asInteger());
            ctx->popData();
            break;

         case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
            op1->setNumeric( _cpr::operaten(op1->asInteger(), op2->asNumeric()) );
            ctx->popData();
            break;
         case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
            op1->content.data.number = _cpr::operaten(op1->asNumeric(), op2->asInteger());
            ctx->popData();
            break;
         case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
            op1->content.data.number = _cpr::operaten(op1->asNumeric(), op2->asNumeric());
            ctx->popData();
            break;

         case FLC_ITEM_USER << 8 | FLC_ITEM_NIL:
         case FLC_ITEM_USER << 8 | FLC_ITEM_BOOL:
         case FLC_ITEM_USER << 8 | FLC_ITEM_INT:
         case FLC_ITEM_USER << 8 | FLC_ITEM_NUM:
         case FLC_ITEM_USER << 8 | FLC_ITEM_METHOD:
         case FLC_ITEM_USER << 8 | FLC_ITEM_FUNC:
         case FLC_ITEM_USER << 8 | FLC_ITEM_USER:
            _cpr::operate( ctx, op1->asClass(), op1->asInst() );
            break;

         default:
            // no need to throw, we're going to get back in the VM.
            throw
               new OperandError( ErrorParam(e_invalid_op, __LINE__, SRC )
                  .origin( ErrorParam::e_orig_vm )
                  .extra(((ExprMath*)ps)->name()) );
         }
      }
      
      // might have gone deep
      if( &cf != &ctx->currentCode() )
      {
         return;
      }
      
      // fallthrough
   
      // Phase 3 -- assigning the topmost value back.
   case 3:
      cf.m_seqId = 4;
      // now assign the topmost item in the stack to the lvalue of self.
      PStep* lvalue = self->first()->lvalueStep();
      if( lvalue != 0 )
      {
         if( ctx->stepInYield( lvalue, cf ) )
         {
            return;
         }
      }
      
   }
   
   // we're done and won't be back.
   ctx->popCode();

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


template
void generic_apply_<ExprAutoPlus::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprAutoMinus::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprAutoTimes::ops>( const PStep* ps, VMContext* ctx);

template
void generic_apply_<ExprAutoDiv::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprAutoMod::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprAutoPow::ops>( const PStep* ps, VMContext* ctx );

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


void ExprMath::describeTo( String& ret ) const
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
   ExprMath( op1, op2, t_minus, "-" )
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
   ExprMath( op1, op2, t_times, "*" )
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
   ExprMath( op1, op2, t_divide, "/" )
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
   ExprMath( op1, op2, t_modulo, "%" )
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
// EXPR LShift
//
ExprRShift::ExprRShift( Expression* op1, Expression* op2 ):
   ExprMath( op1, op2, t_shr, ">>" )
{
   apply = &generic_apply_<ops>;
}

ExprRShift::~ExprRShift()
{}

bool ExprRShift::simplify( Item& value ) const
{
   return generic_simplify<ops>( value, m_first, m_second );
}


//========================================================
// EXPR LShift
//
ExprLShift::ExprLShift( Expression* op1, Expression* op2 ):
   ExprMath( op1, op2, t_shl, "<<" )
{
   apply = &generic_apply_<ops>;
}

ExprLShift::~ExprLShift()
{}

bool ExprLShift::simplify( Item& value ) const
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

//========================================================
// EXPR Bitwise and
//
ExprBAND::ExprBAND( Expression* op1, Expression* op2 ):
   ExprMath( op1, op2, t_band, "^&" )
{
   apply = &generic_apply_<ops>;
}

bool ExprBAND::simplify( Item& value ) const
{
   return generic_simplify<ops>( value, m_first, m_second );
}

//========================================================
// EXPR Bitwise or
//
ExprBOR::ExprBOR( Expression* op1, Expression* op2 ):
   ExprMath( op1, op2, t_bor, "^|" )
{
   apply = &generic_apply_<ops>;
}

bool ExprBOR::simplify( Item& value ) const
{
   return generic_simplify<ops>( value, m_first, m_second );
}


//========================================================
// EXPR Bitwise xor
//
ExprBXOR::ExprBXOR( Expression* op1, Expression* op2 ):
   ExprMath( op1, op2, t_bxor, "^^" )
{
   apply = &generic_apply_<ops>;
}

bool ExprBXOR::simplify( Item& value ) const
{
   return generic_simplify<ops>( value, m_first, m_second );
}


//========================================================
// Auto expressions AAdd
//

ExprAuto::ExprAuto( Expression* op1, Expression* op2, Expression::operator_t t, const String& name ):
   ExprMath( op1, op2, t, name )
{}

//========================================================
// EXPR AAdd
//
ExprAutoPlus::ExprAutoPlus( Expression* op1, Expression* op2 ):
   ExprAuto( op1, op2, t_aadd, "+=" )
{
   apply = &generic_apply_<ops>;
}

ExprAutoPlus::~ExprAutoPlus()
{}

//========================================================
// EXPR ASub
//
ExprAutoMinus::ExprAutoMinus( Expression* op1, Expression* op2 ):
   ExprAuto( op1, op2, t_asub, "-=" )
{
   apply = &generic_apply_<ops>;
}

ExprAutoMinus::~ExprAutoMinus()
{}


//========================================================
// EXPR ATimes
//
ExprAutoTimes::ExprAutoTimes( Expression* op1, Expression* op2 ):
   ExprAuto( op1, op2, t_amul, "*=" )
{
   apply = &generic_apply_<ops>;
}

ExprAutoTimes::~ExprAutoTimes()
{}

//========================================================
// EXPR ADiv
//
ExprAutoDiv::ExprAutoDiv( Expression* op1, Expression* op2 ):
   ExprAuto( op1, op2, t_adiv, "/=" )
{
   apply = &generic_apply_<ops>;
}

ExprAutoDiv::~ExprAutoDiv()
{}

//========================================================
// EXPR AMod
//
ExprAutoMod::ExprAutoMod( Expression* op1, Expression* op2 ):
   ExprAuto( op1, op2, t_amod, "%=" )
{
   apply = &generic_apply_<ops>;
}

ExprAutoMod::~ExprAutoMod()
{}

//========================================================
// EXPR Apow
//
ExprAutoPow::ExprAutoPow( Expression* op1, Expression* op2 ):
   ExprAuto( op1, op2, t_apow, "**=" )
{
   apply = &generic_apply_<ops>;
}

ExprAutoPow::~ExprAutoPow()
{}

//========================================================
// EXPR AShr
//
ExprAutoRShift::ExprAutoRShift( Expression* op1, Expression* op2 ):
   ExprAuto( op1, op2, t_ashr, ">>=" )
{
   apply = &generic_apply_<ops>;
}

ExprAutoRShift::~ExprAutoRShift()
{}

//========================================================
// EXPR AMod

ExprAutoLShift::ExprAutoLShift( Expression* op1, Expression* op2 ):
   ExprAuto( op1, op2, t_shl, "<<=" )
{
   apply = &generic_apply_<ops>;
}

ExprAutoLShift::~ExprAutoLShift()
{}


}

/* end of exprmath.cpp */
