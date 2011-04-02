/*
   FALCON - The Falcon Programming Language.
   FILE: interruptederror.cpp

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/interruptederror.h>
#include <falcon/errorclass.h>
#include <falcon/engine.h>

namespace Falcon {

InterruptedError::InterruptedError( const ErrorParam &params ):
   Error( Engine::instance()->ioErrorClass(), params )
{
}

InterruptedError::~InterruptedError()
{
}

}

/* end of interruptederror.cpp */

