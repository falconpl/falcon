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
#include <falcon/stderrors.h>

#include <falcon/psteps/exprmath.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <math.h>

namespace Falcon {



ExprMath::ExprMath( Expression* op1, Expression* op2, int line, int chr ):
   BinaryExpression( op1, op2, line, chr )
{}

ExprMath::ExprMath( int line, int chr ):
   BinaryExpression( line, chr )
{}

ExprMath::ExprMath( const ExprMath& other ):
   BinaryExpression( other )
{}

ExprAuto::ExprAuto( int line, int chr ):
   ExprMath( line, chr )
{}


ExprAuto::ExprAuto( Expression* op1, Expression* op2, int line, int chr ):
   ExprMath( op1, op2,  line, chr )
{}


ExprAuto::ExprAuto( const ExprAuto& other ):
   ExprMath(other)
{}


class ExprPlus::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a + b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_add(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a + b; }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item& , Item&  ) {}   
   static void assign( PStep*, VMContext* ) {}
};

class ExprMinus::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a - b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_sub(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a - b; }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item&  ) {}   
   static void assign( PStep*, VMContext* ) {}
};

class ExprTimes::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a * b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_mul(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a * b; }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& ) {}   
   static void assign( PStep*, VMContext* ) {}
};

class ExprDiv::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a / b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_div(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a / b; }
   static bool zeroCheck( const Item& n ) { return (n.isInteger() && n.asInteger() == 0) || (n.isNumeric() && n.asNumeric() == 0.0); }
   static void swapper( Item&, Item& op1) { 
      if( op1.isInteger() ) {op1.setNumeric( (numeric) op1.asInteger());} 
   }   
   static void assign( PStep*, VMContext* ) {}
};

class ExprMod::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a % b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_mod(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return (numeric)(((int64)a) % ((int64)b)); }
   static bool zeroCheck( const Item& n ) { return n.isOrdinal() && n.forceInteger() == 0; }
   static void swapper( Item&, Item& ) {}   
   static void assign( PStep*, VMContext* ) {}
};

class ExprPow::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return (int64)pow((numeric)a,(numeric)b); }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_pow(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return pow(a,b); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& op1) { 
      if( op1.isInteger() ) {op1.setNumeric( (numeric) op1.asInteger());} 
   }   
   static void assign( PStep*, VMContext* ) {}
};


class ExprRShift::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return ((uint64)a) >> ((uint64)b); }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_shr(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return (numeric)(((uint64)a) >> ((uint64)b)); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item&) {}
   static void assign( PStep*, VMContext* ) {}
};

class ExprLShift::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return (int64) (((uint64)a) << ((uint64)b)); }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_shl(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return (numeric)(((uint64)a) << ((uint64)b)); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& ) {} 
   static void assign( PStep*, VMContext* ) {}
};


class ExprBAND::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return ((uint64)a) & ((uint64)b); }
   static void operate( VMContext*, Class*, void* ) { 
      throw new OperandError( ErrorParam( e_invop, __LINE__, SRC )
         .extra("^&")
         .origin( ErrorParam::e_orig_vm )); 
   }
   static numeric operaten( numeric a, numeric b ) { return (numeric)(((uint64)a) & ((uint64)b)); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& ) {} 
   static void assign( PStep*, VMContext* ) {}
};

class ExprBOR::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return ((uint64)a) | ((uint64)b); }
   static void operate( VMContext*, Class*, void* ) { 
      throw new OperandError( ErrorParam( e_invop, __LINE__, SRC )
         .extra("^|")
         .origin( ErrorParam::e_orig_vm )); 
   }
   static numeric operaten( numeric a, numeric b ) { return (numeric)(((uint64)a) | ((uint64)b)); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& ) {} 
   static void assign( PStep*, VMContext* ) {}
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
   static numeric operaten( numeric a, numeric b ) { return (numeric)(((uint64)a) ^ ((uint64)b)); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& ) {} 
   static void assign( PStep*, VMContext* ) {}
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
   static void swapper( Item& , Item&  ) {  }
   static void assign( PStep* lvalue, VMContext* ctx ) {if( lvalue != 0 ) { ctx->stepIn( lvalue ); } }
};

class ExprAutoMinus::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a - b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_asub(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a - b; }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& ) { }
   static void assign( PStep* lvalue, VMContext* ctx ) {if( lvalue != 0 ) { ctx->stepIn( lvalue ); } }
};

class ExprAutoTimes::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a * b; }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_amul(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a * b; }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item& , Item&  ) {}
   static void assign( PStep* lvalue, VMContext* ctx ) {if( lvalue != 0 ) { ctx->stepIn( lvalue ); } }
};

class ExprAutoDiv::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a / b; }   
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_adiv(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return a / b; }
   static bool zeroCheck( const Item& n ) { return n.isOrdinal() && n.forceInteger() == 0; }
   static void swapper( Item&, Item& op2 ) {
      if( op2.isInteger() ) {op2.setNumeric( (numeric)op2.asInteger());} 
   }   
   static void assign( PStep* lvalue, VMContext* ctx ) {if( lvalue != 0 ) { ctx->stepIn( lvalue ); } }
};

class ExprAutoMod::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return a % b; }   
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_amod(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return (numeric)(((int64)a) % ((int64)b)); }
   static bool zeroCheck( const Item& n ) { return n.isOrdinal() && n.forceInteger() == 0; }
   static void swapper( Item&, Item& ) { }
   static void assign( PStep* lvalue, VMContext* ctx ) {if( lvalue != 0 ) { ctx->stepIn( lvalue ); } }
};

class ExprAutoPow::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return (int64)pow((numeric)a,(numeric)b); }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_apow(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return pow(a,b); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& op2 ) {
      if( op2.isInteger() ) {op2.setNumeric( (numeric)op2.asInteger());}
   }
   static void assign( PStep* lvalue, VMContext* ctx ) {if( lvalue != 0 ) { ctx->stepIn( lvalue ); } }
};

class ExprAutoRShift::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return ((uint64)a) >> ((uint64)b); }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_ashr(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return (numeric)(((uint64)a) >> ((uint64)b)); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item& , Item&  ) {}
   static void assign( PStep* lvalue, VMContext* ctx ) {if( lvalue != 0 ) { ctx->stepIn( lvalue ); } }
};

class ExprAutoLShift::ops
{
public:
   static int64 operate( int64 a, int64 b ) { return ((uint64)a) << ((uint64)b); }
   static void operate( VMContext* ctx, Class* cls, void* inst ) { cls->op_ashl(ctx, inst); }
   static numeric operaten( numeric a, numeric b ) { return (numeric)(((uint64)a) << ((uint64)b)); }
   static bool zeroCheck( const Item& ) { return false; }
   static void swapper( Item&, Item& ) { }
   static void assign( PStep* lvalue, VMContext* ctx ) {if( lvalue != 0 ) { ctx->stepIn( lvalue ); } }
};



// Inline class to simplify
template <class _cpr >
bool generic_simplify( Item& value, TreeStep* m_first, TreeStep* m_second )
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
   
   fassert( self->first() != 0 );
   fassert( self->second() != 0 );
   
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
      /* no break */
      
      // Phase 1 -- generate the other item.
   case 1:
      cf.m_seqId = 2;
      if( ctx->stepInYield( self->second(), cf ) )
      {
         return;
      }
      /* no break */
   
      // Phase 2 -- operate
   case 2:
      cf.m_seqId = 3;
      {         
         // No need to copy the second, we're not packing the stack now.
         register Item *op1 = &ctx->opcodeParam(1); 
         register Item *op2 = &ctx->opcodeParam(0);
         _cpr::swapper( *op1, *op2 );   

         // we dereference also op1 to help copy-on-write decisions from overrides
         switch ( op1->type() << 8 | op2->type() )
         {
         case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
            if ( _cpr::zeroCheck(*op2) )
            {
               throw new MathError( ErrorParam(e_div_by_zero, __LINE__, SRC )
                  .origin(ErrorParam::e_orig_vm) );
            }
            op1->content.data.val64 = _cpr::operate(op1->asInteger(), op2->asInteger());
            ctx->popData();
            break;

         case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
            if ( _cpr::zeroCheck(*op2) )
            {
               throw new MathError( ErrorParam(e_div_by_zero, __LINE__, SRC )
                  .origin(ErrorParam::e_orig_vm) );
            }
            op1->setNumeric( _cpr::operaten((numeric)op1->asInteger(), op2->asNumeric()) );
            ctx->popData();
            break;
         case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
            if ( _cpr::zeroCheck(*op2) )
            {
               throw new MathError( ErrorParam(e_div_by_zero, __LINE__, SRC )
                  .origin(ErrorParam::e_orig_vm) );
            }
            op1->content.data.number = _cpr::operaten(op1->asNumeric(), (numeric)op2->asInteger());
            ctx->popData();
            break;
         case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
            if ( _cpr::zeroCheck(*op2) )
            {
               throw new MathError( ErrorParam(e_div_by_zero, __LINE__, SRC )
                  .origin(ErrorParam::e_orig_vm) );
            }
            op1->content.data.number = _cpr::operaten(op1->asNumeric(), op2->asNumeric());
            ctx->popData();
            break;

         default:
            {
               // deep items do not require to copy them back.
               ctx->popCode();

               Class* cls = 0;
               void* inst = 0;
               op1->forceClassInst( cls, inst );
               try {
                  _cpr::operate( ctx, cls, inst );
               }
               catch(Error* e)
               {
                  e->line(ps->line());
                  throw e;
               }
               return;
            }
         }

         // might have gone deep
         if( &cf != &ctx->currentCode() )
         {
            return;
         }
      }
      /* no break */
   
      // Phase 3 -- assigning the topmost value back.
   case 3:
      // we're done and won't be back.
      ctx->popCode();
      // however, eventually assign the that we have on the stack back.
      _cpr::assign( self->first()->lvalueStep(), ctx );      
      /* no break */
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

template
void generic_apply_<ExprLShift::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprRShift::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprBAND::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprBOR::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprBXOR::ops>( const PStep* ps, VMContext* ctx );

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

template
void generic_apply_<ExprAutoLShift::ops>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprAutoRShift::ops>( const PStep* ps, VMContext* ctx );

//========================================================
// Implementation
//

#define FALCON_IMPLEMENT_MATH_EXPR_CLASS( name, symbol, handler ) \
   name::name( Expression* op1, Expression* op2, int line, int chr ): \
      ExprMath( op1, op2, line, chr )\
      { FALCON_DECLARE_SYN_CLASS( handler ); apply = &generic_apply_<ops>; }\
   name::name( int line, int chr ): \
      ExprMath( line, chr ) \
      { FALCON_DECLARE_SYN_CLASS( handler ); apply = &generic_apply_<ops>; }\
   name::name( const name &other ): \
      ExprMath( other ) \
      { apply = &generic_apply_<ops>; }\
   bool name::simplify( Item& value ) const {\
      return generic_simplify<ops>( value, m_first, m_second );\
   }\
   const String& name::exprName() const {\
      static String name(symbol);\
      return name;\
   }


FALCON_IMPLEMENT_MATH_EXPR_CLASS( ExprPlus, "+", expr_plus )
FALCON_IMPLEMENT_MATH_EXPR_CLASS( ExprMinus, "-", expr_minus)
FALCON_IMPLEMENT_MATH_EXPR_CLASS( ExprTimes, "*", expr_times )
FALCON_IMPLEMENT_MATH_EXPR_CLASS( ExprDiv, "/", expr_div )
FALCON_IMPLEMENT_MATH_EXPR_CLASS( ExprPow, "**", expr_pow )
FALCON_IMPLEMENT_MATH_EXPR_CLASS( ExprMod, "%", expr_mod )
FALCON_IMPLEMENT_MATH_EXPR_CLASS( ExprLShift, "<<", expr_lshift )
FALCON_IMPLEMENT_MATH_EXPR_CLASS( ExprRShift, ">>", expr_rshift )
FALCON_IMPLEMENT_MATH_EXPR_CLASS( ExprBAND, "^&", expr_band )
FALCON_IMPLEMENT_MATH_EXPR_CLASS( ExprBOR, "^|", expr_bor )
FALCON_IMPLEMENT_MATH_EXPR_CLASS( ExprBXOR, "^^", expr_plus )


#define FALCON_IMPLEMENT_MATH_AUTOEXPR_CLASS( name, symbol, handler ) \
   name::name( Expression* op1, Expression* op2, int line, int chr ): \
      ExprAuto( op1, op2, line, chr )\
      { FALCON_DECLARE_SYN_CLASS( handler ); apply = &generic_apply_<ops>; }\
   name::name( int line, int chr ): \
      ExprAuto( line, chr )\
      { FALCON_DECLARE_SYN_CLASS( handler ); apply = &generic_apply_<ops>; }\
   name::name( const name &other ): \
      ExprAuto( other ) \
      { apply = &generic_apply_<ops>; }\
   const String& name::exprName() const {\
      static String name(symbol);\
      return name;\
   }


FALCON_IMPLEMENT_MATH_AUTOEXPR_CLASS( ExprAutoPlus, "+=", expr_aplus )
FALCON_IMPLEMENT_MATH_AUTOEXPR_CLASS( ExprAutoMinus, "-=", expr_aminus )
FALCON_IMPLEMENT_MATH_AUTOEXPR_CLASS( ExprAutoTimes, "*=", expr_atimes )
FALCON_IMPLEMENT_MATH_AUTOEXPR_CLASS( ExprAutoDiv, "/=", expr_adiv )
FALCON_IMPLEMENT_MATH_AUTOEXPR_CLASS( ExprAutoPow, "**=", expr_apow )
FALCON_IMPLEMENT_MATH_AUTOEXPR_CLASS( ExprAutoMod, "%=", expr_amod )
FALCON_IMPLEMENT_MATH_AUTOEXPR_CLASS( ExprAutoLShift, "<<=", expr_alshift )
FALCON_IMPLEMENT_MATH_AUTOEXPR_CLASS( ExprAutoRShift, ">>=", expr_arshift )

}

/* end of exprmath.cpp */
