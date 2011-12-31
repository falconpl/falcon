/*
   FALCON - The Falcon Programming Language.
   FILE: exprmath.h

   Expression elements -- math ops (very similar and tedouis code)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 23:35:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EXPRMATH_H
#define _FALCON_EXPRMATH_H

#include <falcon/expression.h>

namespace Falcon {

class FALCON_DYN_CLASS ExprMath: public BinaryExpression
{
public:
   inline virtual ~ExprMath() {}

   virtual void describeTo( String&, int depth=0 ) const;

   inline virtual bool isStandAlone() const { return false; }
   virtual bool isStatic() const { return false; }

   const String& name() const { return m_name; }

protected:
   String m_name;
   
   ExprMath( Expression* op1, Expression* op2, const String& name, int line = 0, int chr = 0 );
   ExprMath( const String& name, int line = 0, int chr = 0 );
   ExprMath( const ExprMath& other );
};

#define FALCON_DECLARE_MATH_EXPR_CLASS( ClassName)\
      class FALCON_DYN_CLASS ClassName: public ExprMath\
      {\
      public:\
         ClassName( int line = 0, int chr = 0 );\
         ClassName( Expression* op1, Expression* op2, int line = 0, int chr = 0 );\
         ClassName( const ClassName &other );\
         inline virtual ~ClassName() {}; \
         inline virtual ClassName* clone() const { return new ClassName( *this ); }\
         virtual bool simplify( Item& value ) const; \
      private:\
         class ops;\
      };\

/** Plus opertor. */
FALCON_DECLARE_MATH_EXPR_CLASS( ExprPlus )
/** Minus operator. */
FALCON_DECLARE_MATH_EXPR_CLASS( ExprMinus )
/** Times operator. */
FALCON_DECLARE_MATH_EXPR_CLASS( ExprTimes )
/** Division operator. */
FALCON_DECLARE_MATH_EXPR_CLASS( ExprDiv )
/** Power operator. */
FALCON_DECLARE_MATH_EXPR_CLASS( ExprPow )
/** Expr modulo. */
FALCON_DECLARE_MATH_EXPR_CLASS( ExprMod )


/** Expr shift right */
FALCON_DECLARE_MATH_EXPR_CLASS( ExprRShift )
/** Expr shift left */
FALCON_DECLARE_MATH_EXPR_CLASS( ExprLShift )
/** Expr bitwise and */
FALCON_DECLARE_MATH_EXPR_CLASS( ExprBAND )
/** Expr bitwise or */
FALCON_DECLARE_MATH_EXPR_CLASS( ExprBOR )
/** Expr bitwise or */
FALCON_DECLARE_MATH_EXPR_CLASS( ExprBXOR )


//=====================================================================
// Auto expressions
//
class FALCON_DYN_CLASS ExprAuto: public ExprMath
{
public:   
   inline virtual ~ExprAuto() {}
   virtual bool simplify( Item& ) const { return false; }
   
protected:
   ExprAuto( Expression* op1, Expression* op2, const String& name, int line = 0, int chr = 0 );
   ExprAuto( const String& name, int line = 0, int chr = 0 );
   ExprAuto( const ExprAuto& other );
};

#define FALCON_DECLARE_MATH_AUTOEXPR_CLASS(ClassName)\
      class FALCON_DYN_CLASS ClassName: public ExprAuto\
      {\
      public:\
         ClassName( int line = 0, int chr = 0 );\
         ClassName( Expression* op1, Expression* op2, int line = 0, int chr = 0 );\
         ClassName( const ClassName &other );\
         inline virtual ~ClassName() {}; \
         inline virtual ClassName* clone() const { return new ClassName( *this ); }\
      private:\
         class ops;\
      };\

/** Autp-Plus opertor. */
FALCON_DECLARE_MATH_AUTOEXPR_CLASS( ExprAutoPlus )
/** Auto-Minus operator. */
FALCON_DECLARE_MATH_AUTOEXPR_CLASS( ExprAutoMinus )
/** Auto-Times operator. */
FALCON_DECLARE_MATH_AUTOEXPR_CLASS( ExprAutoTimes )
/** Auto-Division operator. */
FALCON_DECLARE_MATH_AUTOEXPR_CLASS( ExprAutoDiv )
/** Auto-Power operator. */
FALCON_DECLARE_MATH_AUTOEXPR_CLASS( ExprAutoPow )
/** Auto-modulo operator. */
FALCON_DECLARE_MATH_AUTOEXPR_CLASS( ExprAutoMod )
/** Auto-Right shift operator. */
FALCON_DECLARE_MATH_AUTOEXPR_CLASS( ExprAutoRShift )
/** Auto-Left shift operator. */
FALCON_DECLARE_MATH_AUTOEXPR_CLASS( ExprAutoLShift )

}

#endif	/* _FALCON_EXPRMATH_H */

/* end of exprmath.h */
