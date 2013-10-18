/*
   FALCON - The Falcon Programming Language.
   FILE: exprnamed.h

   Syntactic tree item definitions -- Named expression
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Dec 2011 13:22:21 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRNAMED_H
#define FALCON_EXPRNAMED_H

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon {

/** Named expression a|... */
class FALCON_DYN_CLASS ExprNamed: public UnaryExpression
{
public:
   ExprNamed( const String& name, Expression* op1, int line=0, int chr=0 );
   ExprNamed( int line=0, int chr=0 );
   ExprNamed( const ExprNamed& other );
   virtual ~ExprNamed();
   inline virtual ExprNamed* clone() const { return new ExprNamed( *this ); }
   virtual bool simplify( Item& value ) const;
   static void apply_( const PStep*, VMContext* ctx );
   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual const String& exprName() const;
   const String& name() const { return m_name; }
   void name( const String& prop ) { m_name = prop; }

private:
   String m_name;
};

}

#endif	/* EXPRNAME_H */

/* end of exprname.h */
