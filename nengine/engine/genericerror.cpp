/*
   FALCON - The Falcon Programming Language.
   FILE: genericerror.cpp

   Generic Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/engine.h>
#include <falcon/genericerror.h>
#include <falcon/stderrors.h>

namespace Falcon {

GenericError::GenericError( const ErrorParam &params ):
   Error( Engine::instance()->stdErrors()->generic(), params )
{
}

GenericError::~GenericError()
{
}

}

/* end of genericerror.cpp */
