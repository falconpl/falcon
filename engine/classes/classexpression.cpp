/*
   FALCON - The Falcon Programming Language.
   FILE: classexpression.cpp

   Base class for expression PStep handlers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 27 Dec 2011 21:39:56 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/classes/classexpression.h>

namespace Falcon {

ClassExpression::ClassExpression( ClassTreeStep* parent ):
   DerivedFrom( parent, "Expression" )
{}
   
ClassExpression::~ClassExpression(){}

}

/* end of classexpression.cpp */
