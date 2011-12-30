/*
   FALCON - The Falcon Programming Language.
   FILE: exprvector_private.h

   Inner structure holding a vector of expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 18:38:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRVECTOR_PRIVATE_H
#define FALCON_EXPRVECTOR_PRIVATE_H

#include <falcon/expression.h>
#include <vector>

namespace Falcon
{

class ExprVector_Private {
public:

   typedef std::vector< Expression* > ExprVector;
   ExprVector m_exprs;

   ~ExprVector_Private()
   {
      ExprVector::iterator iter = m_exprs.begin();
      while( iter != m_exprs.end() )
      {
         delete (*iter);
         ++iter;
      }
   }
};
}

#endif

/* end of exprvector_private.h */
