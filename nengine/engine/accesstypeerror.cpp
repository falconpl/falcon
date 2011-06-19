/*
   FALCON - The Falcon Programming Language.
   FILE: accesstypeerror.cpp

    Error while accessing objects in read/write mode.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jun 2011 15:56:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/engine.h>
#include <falcon/accesstypeerror.h>
#include <falcon/errorclass.h>

namespace Falcon {

AccessTypeError::AccessTypeError( const ErrorParam &params ):
   Error( Engine::instance()->accessTypeErrorClass(), params )
{
}

AccessTypeError::~AccessTypeError()
{
}

}

/* end of accesstypeerror.cpp */
