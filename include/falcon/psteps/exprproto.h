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

#include <falcon/psteps/exprvector.h>

namespace Falcon
{

/** Expression generating a prototype instance.

 This expression is equivalent to Prototype() call follwed by
 all the assignments that it declares.

   /TODO: Homoiconicity requires a review.
 */
class FALCON_DYN_CLASS ExprProto: public Expression
{
public:
   ExprProto(int line = 0, int chr = 0);
   ExprProto( const ExprProto& other );
   virtual ~ExprProto();

   size_t size() const;

   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;   
   virtual bool setNth( int32 n, TreeStep* ts );
   virtual bool remove( int n );
   //virtual bool append( TreeStep* ts );
   
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

   static void apply_( const PStep*, VMContext* ctx );

   inline virtual ExprProto* clone() const { return new ExprProto( *this ); }

   virtual bool isStatic() const { return false; }
   virtual bool simplify( Item& result ) const;
   virtual void render( TextWriter* tw, int32 depth ) const;
   
private:
   class Private;
   Private* _p;
};

}

#endif

/* end of exprproto.h */
