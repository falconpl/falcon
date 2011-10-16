/*
   FALCON - The Falcon Programming Language.
   FILE: exprproto.h

   Syntactic tree item definitions -- prototype generator.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 16:55:07 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRPROTO_H
#define FALCON_EXPRPROTO_H

#include <falcon/expression.h>

namespace Falcon
{

/** Expression generating a prototype instance.

 This expression is equivalent to Prototype() call follwed by
 all the assignments that it declares.

 */
class FALCON_DYN_CLASS ExprProto: public Expression
{
public:
   ExprProto();
   ExprProto( const ExprProto& other );
   virtual ~ExprProto();

   size_t size() const;

   /** Get the nth expression in the declaration.
    \param n The number of the expression that must be accessed.
    \return the nth expression or 0 if n is out of range.
    */
   Expression* exprAt( size_t n ) const;

   /** Get the nth name in the declaration.
    \param n The number of the expression that must be accessed.
    \return the nth expression or 0 if n is out of range.
    */
   const String& nameAt( size_t n ) const;

   /** Adds another expression to this array.
    \return itself (useful for declarations in sources)
    */
   ExprProto& add( const String& name, Expression* e );

   virtual void serialize( DataWriter* s ) const;
   virtual void precompile( PCode* pcd ) const;

   virtual void describeTo( String& ) const;

   static void apply_( const PStep*, VMContext* ctx );

   inline virtual ExprProto* clone() const { return new ExprProto( *this ); }

   virtual bool isStatic() const { return false; }
   virtual bool simplify( Item& result ) const;

protected:
   virtual void deserialize( DataReader* s );

private:
   class Private;
   Private* _p;
};

}

#endif

/* end of exprproto.h */
