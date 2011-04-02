/*
   FALCON - The Falcon Programming Language.
   FILE: unsupportederror.cpp

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/unsupportederror.h>
#include <falcon/errorclass.h>
#include <falcon/engine.h>

namespace Falcon {

UnsupportedError::UnsupportedError( const ErrorParam &params ):
   Error( Engine::instance()->unsupportedErrorClass(), params )
{
}

UnsupportedError::~UnsupportedError()
{
}

}

/* end of unsupportederror.cpp */
