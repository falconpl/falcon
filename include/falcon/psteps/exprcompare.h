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

/** Abstract base class for comparers. */
class FALCON_DYN_CLASS ExprCompare: public BinaryExpression
{
public:
   virtual ~ExprCompare();

   inline virtual bool isStandAlone() const { return false; }

   virtual bool checkCompare( int64 compresult ) const = 0;

   class FALCON_DYN_CLASS PStepPostCompare: public PStep
   {
   public:
      PStepPostCompare(ExprCompare* owner): m_owner(owner) { apply = apply_; }
      virtual ~PStepPostCompare() {}
      virtual void describeTo( String& str ) const { str = "PStepPostCompare"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      ExprCompare* m_owner;
   };

   PStepPostCompare m_stepPostComparer;

protected:
   ExprCompare( int line = 0, int chr = 0 );
   ExprCompare( Expression* op1, Expression* op2, int line = 0, int chr = 0 );
   ExprCompare( const ExprCompare& other );
};


/** Less than operator. */
class FALCON_DYN_CLASS ExprLT: public ExprCompare
{
public:
   ExprLT( int line = 0, int chr = 0 );
   ExprLT( Expression* op1, Expression* op2, int line = 0, int chr = 0 );
   ExprLT( const ExprLT& other );

   virtual ~ExprLT();
   inline virtual ExprLT* clone() const { return new ExprLT( *this ); }
   virtual bool simplify( Item& value ) const;

   virtual bool checkCompare( int64 value ) const { return value < 0; }
   virtual const String& exprName() const;

   class FALCON_DYN_CLASS comparer
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
   ExprLE( int line = 0, int chr = 0 );
   ExprLE( Expression* op1, Expression* op2, int line = 0, int chr = 0 );
   ExprLE( const ExprLT& other );

   virtual ~ExprLE();

   inline virtual ExprLE* clone() const { return new ExprLE( *this ); }

   virtual bool simplify( Item& value ) const;
   virtual bool checkCompare( int64 value ) const { return value <= 0; }
   virtual const String& exprName() const;

   class FALCON_DYN_CLASS comparer
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
   ExprGT( int line = 0, int chr = 0 );
   ExprGT( Expression* op1, Expression* op2, int line = 0, int chr = 0 );
   ExprGT( const ExprGT& other );

   virtual ~ExprGT();
   inline virtual ExprGT* clone() const { return new ExprGT( *this ); }

   virtual bool simplify( Item& value ) const;

   virtual bool checkCompare( int64 value ) const { return value > 0; }
   virtual const String& exprName() const;

   class FALCON_DYN_CLASS comparer
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
   ExprGE( int line = 0, int chr = 0 );
   ExprGE( Expression* op1, Expression* op2, int line = 0, int chr = 0 );
   ExprGE( const ExprGE& other );

   virtual ~ExprGE();
   inline virtual ExprGE* clone() const { return new ExprGE( *this ); }
   virtual bool simplify( Item& value ) const;
   virtual bool checkCompare( int64 value ) const { return value >= 0; }
   virtual const String& exprName() const;

   class FALCON_DYN_CLASS comparer
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
   ExprEQ( int line = 0, int chr = 0 );
   ExprEQ( Expression* op1, Expression* op2, int line = 0, int chr = 0 );
   ExprEQ( const ExprEQ& other );

   virtual ~ExprEQ();
   inline virtual ExprEQ* clone() const { return new ExprEQ( *this ); }
   virtual bool simplify( Item& value ) const;
   virtual bool checkCompare( int64 value ) const { return value == 0; }
   virtual const String& exprName() const;

   class FALCON_DYN_CLASS comparer
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
   ExprNE( int line = 0, int chr = 0 );
   ExprNE( Expression* op1, Expression* op2, int line = 0, int chr = 0 );
   ExprNE( const ExprNE& other );

   virtual ~ExprNE();
   inline virtual ExprNE* clone() const { return new ExprNE( *this ); }
   virtual bool simplify( Item& value ) const;
   virtual bool checkCompare( int64 value ) const { return value != 0; }
   virtual const String& exprName() const;


   class FALCON_DYN_CLASS comparer
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
