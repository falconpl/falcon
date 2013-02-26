/*
   FALCON - The Falcon Programming Language.
   FILE: typeerror.cpp

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 26 Feb 2013 14:26:15 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/errors/typeerror.h>
#include <falcon/engine.h>
#include <falcon/stderrors.h>

namespace Falcon {

TypeError::TypeError( ):
   Error( Engine::instance()->stdErrors()->type() )
{
}

TypeError::TypeError( const ErrorParam &params ):
   Error( Engine::instance()->stdErrors()->type(), params )
{
}

TypeError::~TypeError()
{
}

}

/* end of typeerror.cpp */
