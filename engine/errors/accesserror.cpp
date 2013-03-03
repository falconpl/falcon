/*
   FALCON - The Falcon Programming Language.
   FILE: accesserror.cpp

   Error while accessing objects (with square or dot operator).
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 20:59:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/engine.h>
#include <falcon/errors/accesserror.h>
#include <falcon/stderrors.h>

namespace Falcon {

AccessError::AccessError( ):
   Error( Engine::instance()->stdErrors()->access() )
{
}

AccessError::AccessError( const ErrorParam &params ):
   Error( Engine::instance()->stdErrors()->access(), params )
{
}

AccessError::~AccessError()
{
}

}

/* end of accesserror.cpp */
