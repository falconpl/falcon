/*
   FALCON - The Falcon Programming Language.
   FILE: exprstarindex.cpp

   Syntactic tree item definitions -- Star-index accessor
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRSTARINDEX_H_
#ifndef FALCON_EXPRSTARINDEX_H_

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon {

/** Special string Index accessor. */
class FALCON_DYN_CLASS ExprStarIndex: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprStarIndex, expr_starindex );
};


}

#endif

/* end of exprstarindex.h */
