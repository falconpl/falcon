/*
   FALCON - The Falcon Programming Language.
   FILE: exprcompare.h

   Expression elements -- Comparisons
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 23:35:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EXPRCOMPARE_H
#define	_FALCON_EXPRCOMPARE_H

#include <falcon/expression.h>

namespace Falcon {

class FALCON_DYN_CLASS ExprCompare: public BinaryExpression
{
public:
   ExprCompare( Expression* op1, Expression* op2, operator_t t, const String& name );
   ExprCompare( const ExprCompare& other );
   virtual ~ExprCompare();

   virtual void describeTo( String&, int depth=0 ) const;

   inline virtual bool isStandAlone() const { return false; }
   virtual bool isStatic() const { return false; }

   const String& name() const { return m_name; }

public:
   String m_name;
};


/** Less than operator. */
class FALCON_DYN_CLASS ExprLT: public ExprCompare
{
public:
   ExprLT( Expression* op1=0, Expression* op2=0 );

   ExprLT( const ExprLT& other ):
      ExprCompare(other)
   {}

   virtual ~ExprLT();

   inline virtual ExprLT* clone() const { return new ExprLT( *this ); }
   
   virtual bool simplify( Item& value ) const;

   class comparer
   {
   public:
      static bool pass( int64 a, int64 b ) { return a < b; }
      static bool passn( numeric a, numeric b ) { return a < b; }
      static bool cmpCheck( int64 value ) { return value < 0; }
   };

};

/** Less or equal operator. */
class FALCON_DYN_CLASS ExprLE: public ExprCompare
{
public:
   ExprLE( Expression* op1=0, Expression* op2=0 );

   ExprLE( const ExprLT& other ):
      ExprCompare(other)
   {}

   virtual ~ExprLE();

   inline virtual ExprLE* clone() const { return new ExprLE( *this ); }

   virtual bool simplify( Item& value ) const;

   class comparer
   {
   public:
      static bool pass( int64 a, int64 b ) { return a <= b; }
      static bool passn( numeric a, numeric b ) { return a <= b; }
      static bool cmpCheck( int64 value ) { return value <= 0; }
   };

};

/** Greater than operator. */
class FALCON_DYN_CLASS ExprGT: public ExprCompare
{
public:
   ExprGT( Expression* op1=0, Expression* op2=0 );

   ExprGT( const ExprGT& other ):
      ExprCompare(other)
   {}

   virtual ~ExprGT();
   inline virtual ExprGT* clone() const { return new ExprGT( *this ); }

   virtual bool simplify( Item& value ) const;


   class comparer
   {
   public:
      static bool pass( int64 a, int64 b ) { return a > b; }
      static bool passn( numeric a, numeric b ) { return a > b; }
      static bool cmpCheck( int64 value ) { return value > 0; }
   };

};


/** Greater than operator. */
class FALCON_DYN_CLASS ExprGE: public ExprCompare
{
public:
   ExprGE( Expression* op1=0, Expression* op2=0 );
   
   ExprGE( const ExprGE& other ):
      ExprCompare(other)
   {}

   virtual ~ExprGE();
   inline virtual ExprGE* clone() const { return new ExprGE( *this ); }

   virtual bool simplify( Item& value ) const;

   class comparer
   {
   public:
      static bool pass( int64 a, int64 b ) { return a >= b; }
      static bool passn( numeric a, numeric b ) { return a >= b; }
      static bool cmpCheck( int64 value ) { return value >= 0; }
   };

};


/** Greater than operator. */
class FALCON_DYN_CLASS ExprEQ: public ExprCompare
{
public:
   ExprEQ( Expression* op1=0, Expression* op2=0 );

   ExprEQ( const ExprEQ& other ):
      ExprCompare(other)
   {}

   virtual ~ExprEQ();
   inline virtual ExprEQ* clone() const { return new ExprEQ( *this ); }

   virtual bool simplify( Item& value ) const;

   class comparer
   {
   public:
      static bool pass( int64 a, int64 b ) { return a == b; }
      static bool passn( numeric a, numeric b ) { return a == b; }
      static bool cmpCheck( int64 value ) { return value == 0; }
   };

};


/** Greater than operator. */
class FALCON_DYN_CLASS ExprNE: public ExprCompare
{
public:
   ExprNE( Expression* op1=0, Expression* op2=0 );
   
   ExprNE( const ExprNE& other ):
      ExprCompare(other)
   {}

   virtual ~ExprNE();
   inline virtual ExprNE* clone() const { return new ExprNE( *this ); }

   virtual bool simplify( Item& value ) const;

   class comparer
   {
   public:
      static bool pass( int64 a, int64 b ) { return a != b; }
      static bool passn( numeric a, numeric b ) { return a != b; }
      static bool cmpCheck( int64 value ) { return value != 0; }
   };

};

}

#endif	/* _FALCON_EXPRCOMPARE_H */

/* end of exprcompare.h */
