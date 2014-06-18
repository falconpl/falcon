/*
   FALCON - The Falcon Programming Language.
   FILE: exprclosure.h

   Genearte a closure out of a function value.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Jan 2012 21:15:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRCLOSURE_H
#define FALCON_EXPRCLOSURE_H

#include <falcon/setup.h>
#include <falcon/expression.h>

namespace Falcon {

class Function;

class ExprClosure: public Expression
{
public:
   ExprClosure();
   ExprClosure( Function* closed );
   ExprClosure( const ExprClosure& other );
   ~ExprClosure();
   
   virtual void render( TextWriter* tgt, int32 depth ) const;
   inline virtual ExprClosure* clone() const { return new ExprClosure( *this ); }

   virtual bool simplify( Item& ) const { return false; }
   
   Function* function() const { return m_function; }
   void function( Function* f );
   
private:
   Function* m_function;
   
   static void apply_( const PStep*, VMContext* vm );
};

}

#endif	/* EXPRCLOSURE_H */

/* end of exprclosure.h */
