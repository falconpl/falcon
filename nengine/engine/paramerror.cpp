/*
   FALCON - The Falcon Programming Language.
   FILE: unsupportederror.cpp

   Error class raised when functions receive invalid parameters.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 07 May 2011 19:29:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/paramerror.h>
#include <falcon/errorclass.h>
#include <falcon/engine.h>

namespace Falcon {

ParamError::ParamError( const ErrorParam &params ):
   Error( Engine::instance()->paramErrorClass(), params )
{
}

ParamError::~ParamError()
{
}

}

/* end of unsupportederror.cpp */
