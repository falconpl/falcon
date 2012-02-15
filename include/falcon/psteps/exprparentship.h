/*
   FALCON - The Falcon Programming Language.
   FILE: exprparentship.h

   Structure holding information about inheritance in a class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 12 Feb 2012 21:56:43 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EXPRPARENTSHIP_H_
#define _FALCON_EXPRPARENTSHIP_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/pstep.h>
#include <falcon/sourceref.h>
#include <falcon/requirement.h>
#include <falcon/psteps/exprvector.h>

namespace Falcon
{

class VMContext;

/** Expression holding a list of inheritance expressions. 
 
 This expression invokes the initialization sequence of each
 sub-expression in turn (last to first).
 */
class FALCON_DYN_CLASS ExprParentship: public ExprVector
{
public:
   ExprParentship( int line=0, int chr=0 );  
   ExprParentship( const ExprParentship& other );   
   virtual ~ExprParentship();

   virtual void describeTo( String& target, int depth = 0 ) const;
   
   /** Overridden to filter out non-inheritance expressions. */
   virtual bool setNth( int32 n, TreeStep* ts );
   /** Overridden to filter out non-inheritance expressions. */
   virtual bool insert( int32 pos, TreeStep* element );  

   virtual bool isStatic() const { return false; }
   virtual bool simplify( Item& ) const { return false; }  
   virtual ExprParentship* clone() const { return new ExprParentship(*this); }
   
private:
   
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif /* _FALCON_EXPRPARENTSHIP_H_ */

/* end of exprparentship.h */
