/*
   FALCON - The Falcon Programming Language.
   FILE: exprfactory.h

   Syntactic tree item definitions -- expression factory.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRFACTORY_H
#define FALCON_EXPRFACTORY_H

#include <falcon/expression.h>

namespace Falcon {

/** Expression factory.
 *  Used during de-serialization to create the proper type of expression to be deserialized.
 */
class FALCON_DYN_CLASS ExprFactory {
public:
   static Expression* make( Expression::operator_t type );
   static Expression* deserialize( DataReader* s );
};


}

#endif

/* end of exprfactory.h */
