/*
   FALCON - The Falcon Programming Language.
   FILE: classexpression.h

   Base class for expression PStep handlers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 27 Dec 2011 21:39:56 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSEXPRESSION_H_
#define _FALCON_CLASSEXPRESSION_H_

#include <falcon/setup.h>
#include <falcon/derivedfrom.h>

namespace Falcon {

class ClassTreeStep;

/** Handler class for Expression class.
 */
class ClassExpression: public DerivedFrom // TreeStep
{
public:
   ClassExpression( ClassTreeStep* parent );
   virtual ~ClassExpression();
   void op_call( VMContext* ctx, int pcount, void* self ) const;
};

}

#endif 

/* end of classexpression.h */
