/*
   FALCON - The Falcon Programming Language.
   FILE: exprvector.h

   Common interface for expressions holding vector of expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Dec 2011 12:51:26 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRVECTOR_H
#define FALCON_EXPRVECTOR_H

#include <falcon/expression.h>

namespace Falcon
{

class TreeStepVector_Private;

/** Common interface for expressions holding vector of expressions.
 This is an abstract class; you can't instance it directly
 */
class FALCON_DYN_CLASS ExprVector: public Expression
{
public:      
   virtual ~ExprVector();
   
   virtual int arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   virtual bool insert( int32 pos, TreeStep* element );   
   virtual bool remove( int32 pos );
   virtual bool append( TreeStep* element );

   void resolveUnquote( VMContext* ctx, const UnquoteResolver& );

protected:
   TreeStepVector_Private* _p;
   ExprVector();   
   ExprVector( int line, int chr);
   ExprVector( const ExprVector& other );

};

}

#endif

/* end of exprvector.h */
