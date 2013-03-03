/*
   FALCON - The Falcon Programming Language.
   FILE: exprep.h

   Evaluation Parameters expression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 Jan 2013 13:30:03 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPREP_H
#define FALCON_EXPREP_H

#include <falcon/psteps/exprvector.h>

namespace Falcon
{

/** Evaluation Parameters expression.

   The Evaluation Parameter Expression (EP-ex for short)
   is a literal list of zero to N sub-expressions enclosed in ^(....),
   that are used as parameters in subsequent calls via the indirect
   evaluation operator (#).

   For instance:
   \code
   ep_ex = ^("Hello", " ", "world!" )
   printl # ep_ex  // "Hello world!"
   \endcode

   The ^( ... ) block has the semantics of a literal block; it can host lazily
   evaluated sub-expressions and unescapes as the {()} block.

   \code
   function make_ep( param )
      return ^( a*2, " ", ^~(param*2) )
   end

   a = 10
   ep = make_ep(10)
   > ep            // ^( a*2, " ", 20)
   printl # ep     // 10 20
   \endcode

   Eta functions will receive the unescaped expressions as-is.

   \code
   function *someEta()
      for i in [0:paramCount()]
         > param(i)
      end
   end
   \endcode

   someEta # ^( expr+1, v*2, ^(p,c) )
   // prints:
   //   expr+1
   //   v*2
   //   ^(p,c)
   \endcode
 */
class FALCON_DYN_CLASS ExprEP: public ExprVector
{
public:
   ExprEP( int line=0, int chr=0);
   ExprEP( const ExprEP& other );
   virtual ~ExprEP();

   inline virtual ExprEP* clone() const { return new ExprEP( *this ); }
   virtual bool simplify( Item& value ) const;
   virtual void describeTo( String&, int depth=0 ) const;

   inline virtual bool isStandAlone() const { return false; }

   virtual bool isStatic() const { return false; }

private:
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of exprep.h */
