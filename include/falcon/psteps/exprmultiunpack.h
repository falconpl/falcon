/*
   FALCON - The Falcon Programming Language.
   FILE: exprmultiunpack.h

    Class handling a parallel array of expressions and symbols where to assign them
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Dec 2011 13:22:21 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRMULTIUNPACK_H_
#define FALCON_EXPRMULTIUNPACK_H_

#include <falcon/expression.h>

namespace Falcon {

/** Class handling a parallel array of expressions and symbols where to assign them. 
 * \TODO: Derive this from ExprVector and mask insert/remove away.
 * \TODO: Need to fix homoiconicity
 */
class FALCON_DYN_CLASS ExprMultiUnpack: public Expression
{
public:
   ExprMultiUnpack( int line = 0, int chr = 0 );
   ExprMultiUnpack( bool bIsTop, int line = 0, int chr = 0 );
   ExprMultiUnpack( const ExprMultiUnpack& other );
   virtual ~ExprMultiUnpack();

   inline virtual ExprMultiUnpack* clone() const { return new ExprMultiUnpack( *this ); }
   virtual bool simplify( Item& value ) const;
   virtual void render( TextWriter* tw, int32 depth ) const;
   inline virtual bool isStandAlone() const { return true; }

   int targetCount() const;
   Symbol* getAssignand( int n ) const;
   Expression* getAssignee( int n ) const;
   ExprMultiUnpack& addAssignment( Symbol* tgt, Expression* src );

   virtual bool isStatic() const { return false; }

   bool isTop() const { return m_bIsTop; }

   void resolveUnquote( VMContext* ctx );
protected:
   bool m_bIsTop;

private:
   class Private;
   Private* _p;

   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of exprmultiunpack.h */
