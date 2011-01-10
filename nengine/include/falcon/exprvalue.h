/*
   FALCON - The Falcon Programming Language.
   FILE: exprvalue.h

   Syntactic tree item definitions -- expression elements -- value.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRVALUE_H
#define FALCON_EXPRVALUE_H

#include <falcon/setup.h>
#include <falcon/expression.h>
#include <falcon/item.h>

namespace Falcon {

class Stream;
class GarbageLock;
class VMachine;

/** Expression holding a constant value.
 *
 * This is a value-expression, or an expression that evaluates
 * into a single, constant atomic value.
 */
class FALCON_DYN_CLASS ExprValue: public Expression
{
public:
   ExprValue( const Item& item );

   ExprValue( const ExprValue& other );

   virtual ~ExprValue();


   virtual bool simplify( Item& result ) const;

   virtual void perform( VMachine* vm ) const;
   virtual void apply( VMachine* vm ) const;

   /** Returns a const version of the stored item.
    * The stored item cannot be changed directly; to change it,
    * use the set accessor;
    */
   const Item& item() const { return m_item; }

   /** Changes the stored item.
    */
   void item( const Item& i );

   virtual ExprValue* clone() const;
   virtual void serialize( Stream* s ) const;
   virtual bool isStatic() const;
   virtual const String toString() const;

protected:
   virtual void deserialize( Stream* s );
   inline ExprValue( const Item& item ):
      Expression( t_value ),
      m_lock(0)
   {}

private:
   Item m_item;
   GarbageLock* m_lock;
};

}

#endif

/* end of exprvalue.h */
