/*
   FALCON - The Falcon Programming Language.
   FILE: concurrencyerror.cpp

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Apr 2013 16:25:03 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/errors/concurrencyerror.h>
#include <falcon/engine.h>
#include <falcon/stderrors.h>

namespace Falcon {

ConcurrencyError::ConcurrencyError( ):
   Error( Engine::instance()->stdErrors()->concurrency() )
{
}

ConcurrencyError::ConcurrencyError( const ErrorParam &params ):
   Error( Engine::instance()->stdErrors()->concurrency(), params )
{
}

ConcurrencyError::~ConcurrencyError()
{
}

}

/* end of concurrencyerror.cpp */
