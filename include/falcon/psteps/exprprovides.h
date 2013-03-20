/*
   FALCON - The Falcon Programming Language.
   FILE: exprprovides.h

   Syntactic tree item definitions -- Operator "provides"
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 04 Feb 2013 19:39:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRPROVIDES_H
#define FALCON_EXPRPROVIDES_H

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon {

/** The "provides" operator
 * \TODO: Need to fix homoiconicity
 */
class FALCON_DYN_CLASS ExprProvides: public UnaryExpression
{
public:

   ExprProvides( Expression* op1, const String& property, int line=0, int chr=0 );
   ExprProvides(int line=0, int chr=0);
   ExprProvides( const ExprProvides& other );
   virtual ~ExprProvides();
   inline virtual ExprProvides* clone() const { return new ExprProvides( *this ); }
   virtual bool simplify( Item& value ) const;
   static void apply_( const PStep*, VMContext* ctx );
   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual const String& exprName() const;
   const String& property() const { return m_property; }
   void property( const String& prop ) { m_property = prop; }

private:
   String m_property;
};

}

#endif

/* end of exprprovides.h */
