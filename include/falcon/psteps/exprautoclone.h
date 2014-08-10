/*
   FALCON - The Falcon Programming Language.
   FILE: exprautoclone.h

   Syntactic tree item definitions -- expression elements -- clone.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 08 Apr 2013 16:04:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRAUTOCLONE_H
#define FALCON_EXPRAUTOCLONE_H

#include <falcon/setup.h>
#include <falcon/expression.h>
#include <falcon/item.h>

namespace Falcon {

class Stream;
class VMachine;

/** Expression holding a constant value that is cloned at each invocation.
 *
 * This is a value-expression, or an expression that evaluates
 * into a value that gets cloned each time the expression is evaluated.
 *
 * This entity owns the data, and will destroy it when it is destroyed
 * by using the data handler Class::dispose method.
 */
class FALCON_DYN_CLASS ExprAutoClone: public Expression
{
public:
   ExprAutoClone( int line = 0, int chr = 0 );
   ExprAutoClone( const Class* cls, void* data, int line = 0, int chr = 0 );
   ExprAutoClone( const ExprAutoClone& other );

   virtual ~ExprAutoClone();
   static void apply_( const PStep* s1, VMContext* ctx );

   void set( Class* cls, void* data );
   const Class* cloneHandler() const { return m_cls; }
   void* cloneData() const { return m_data; }

   virtual ExprAutoClone* clone() const;
   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual bool isStandAlone() const { return false; }

private:
   const Class* m_cls;
   void* m_data;
};

}

#endif

/* end of exprautoclone.h */
