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
#define	_FALCON_EXPRMATH_H

#include <falcon/expression.h>

namespace Falcon {

class FALCON_DYN_CLASS ExprMath: public BinaryExpression
{
public:
   ExprMath( Expression* op1, Expression* op2, operator_t t, const String& name );
   ExprMath( const ExprMath& other );
   virtual ~ExprMath();

   virtual void describeTo( String& ) const;

   inline virtual bool isStandAlone() const { return false; }
   virtual bool isStatic() const { return false; }

   const String& name() const { return m_name; }

protected:
   friend class ExprFactory;

public:
   String m_name;
};


/** Plus opertor. */
class FALCON_DYN_CLASS ExprPlus: public ExprMath
{
public:
   ExprPlus( Expression* op1=0, Expression* op2=0 );

   ExprPlus( const ExprPlus& other ):
      ExprMath(other)
   {}

   virtual ~ExprPlus();

   inline virtual ExprPlus* clone() const { return new ExprPlus( *this ); }

   virtual bool simplify( Item& value ) const;   

protected:
   friend class ExprFactory;

private:
   class ops;
};

/** Minus operator. */
class FALCON_DYN_CLASS ExprMinus: public ExprMath
{
public:
   ExprMinus( Expression* op1=0, Expression* op2=0 );

   ExprMinus( const ExprMinus& other ):
      ExprMath(other)
   {}

   virtual ~ExprMinus();

   inline virtual ExprMinus* clone() const { return new ExprMinus( *this ); }

   virtual bool simplify( Item& value ) const;

protected:
   friend class ExprFactory;

private:
   class ops;
};

/** Times operator. */
class FALCON_DYN_CLASS ExprTimes: public ExprMath
{
public:
   ExprTimes( Expression* op1=0, Expression* op2=0 );

   ExprTimes( const ExprTimes& other ):
      ExprMath(other)
   {}

   virtual ~ExprTimes();
   inline virtual ExprTimes* clone() const { return new ExprTimes( *this ); }

   virtual bool simplify( Item& value ) const;

protected:
   friend class ExprFactory;

private:
   class ops;
};


/** Division operator. */
class FALCON_DYN_CLASS ExprDiv: public ExprMath
{
public:
   ExprDiv( Expression* op1=0, Expression* op2=0 );

   ExprDiv( const ExprDiv& other ):
      ExprMath(other)
   {}

   virtual ~ExprDiv();
   inline virtual ExprDiv* clone() const { return new ExprDiv( *this ); }

   virtual bool simplify( Item& value ) const;

protected:
   friend class ExprFactory;

private:
   class ops;
};


/** Power operator. */
class FALCON_DYN_CLASS ExprPow: public ExprMath
{
public:
   ExprPow( Expression* op1=0, Expression* op2=0 );

   ExprPow( const ExprPow& other ):
      ExprMath(other)
   {}

   virtual ~ExprPow();
   inline virtual ExprPow* clone() const { return new ExprPow( *this ); }

   virtual bool simplify( Item& value ) const;

protected:
   friend class ExprFactory;

private:
   class ops;
};


/** Expr modulo. */
class FALCON_DYN_CLASS ExprMod: public ExprMath
{
public:
   ExprMod( Expression* op1=0, Expression* op2=0 );

   ExprMod( const ExprMod& other ):
      ExprMath(other)
   {}

   virtual ~ExprMod();
   inline virtual ExprMod* clone() const { return new ExprMod( *this ); }

   virtual bool simplify( Item& value ) const;

protected:
   friend class ExprFactory;

private:
   class ops;
};

//=====================================================================
// Auto expressions
//
class FALCON_DYN_CLASS ExprAuto: public ExprMath
{
public:
   ExprAuto( Expression* op1, Expression* op2, Expression::operator_t t, const String& name );

   ExprAuto( const ExprAuto& other ):
      ExprMath(other)
   {}
   
   virtual bool simplify( Item& ) const { return false; } 
   virtual void precompile( PCode* pc ) const;
protected:
   friend class ExprFactory;
};

/** Autp-Plus opertor. */
class FALCON_DYN_CLASS ExprAutoPlus: public ExprAuto
{
public:
   ExprAutoPlus( Expression* op1=0, Expression* op2=0 );

   ExprAutoPlus( const ExprAutoPlus& other ):
      ExprAuto(other)
   {}
   virtual ~ExprAutoPlus();
   inline virtual ExprAutoPlus* clone() const { return new ExprAutoPlus( *this ); }
protected:
   friend class ExprFactory;

private:
   class ops;
};

/** Auto-Minus operator. */
class FALCON_DYN_CLASS ExprAutoMinus: public ExprAuto
{
public:
   ExprAutoMinus( Expression* op1=0, Expression* op2=0 );

   ExprAutoMinus( const ExprAutoMinus& other ):
      ExprAuto(other)
   {}

   virtual ~ExprAutoMinus();
   inline virtual ExprAutoMinus* clone() const { return new ExprAutoMinus( *this ); }

protected:
   friend class ExprFactory;

private:
   class ops;
};


/** Auto-Times operator. */
class FALCON_DYN_CLASS ExprAutoTimes: public ExprAuto
{
public:
   ExprAutoTimes( Expression* op1=0, Expression* op2=0 );

   ExprAutoTimes( const ExprAutoTimes& other ):
      ExprAuto(other)
   {}

   virtual ~ExprAutoTimes();
   inline virtual ExprAutoTimes* clone() const { return new ExprAutoTimes( *this ); }

protected:
   friend class ExprFactory;

private:
   class ops;
};


/** Auto-Division operator. */
class FALCON_DYN_CLASS ExprAutoDiv: public ExprAuto
{
public:
   ExprAutoDiv( Expression* op1=0, Expression* op2=0 );

   ExprAutoDiv( const ExprAutoDiv& other ):
      ExprAuto(other)
   {}

   virtual ~ExprAutoDiv();
   inline virtual ExprAutoDiv* clone() const { return new ExprAutoDiv( *this ); }

protected:
   friend class ExprFactory;

private:
   class ops;
};


/** Auto-Power operator. */
class FALCON_DYN_CLASS ExprAutoPow: public ExprAuto
{
public:
   ExprAutoPow( Expression* op1=0, Expression* op2=0 );

   ExprAutoPow( const ExprAutoPow& other ):
      ExprAuto(other)
   {}

   virtual ~ExprAutoPow();
   inline virtual ExprAutoPow* clone() const { return new ExprAutoPow( *this ); }

protected:
   friend class ExprFactory;

private:
   class ops;
};


/** Auto-modulo operator. */
class FALCON_DYN_CLASS ExprAutoMod: public ExprAuto
{
public:
   ExprAutoMod( Expression* op1=0, Expression* op2=0 );

   ExprAutoMod( const ExprAutoMod& other ):
      ExprAuto(other)
   {}

   virtual ~ExprAutoMod();
   inline virtual ExprAutoMod* clone() const { return new ExprAutoMod( *this ); }

protected:
   friend class ExprFactory;

private:
   class ops;
};

}

#endif	/* _FALCON_EXPRMATH_H */

/* end of exprmath.h */
