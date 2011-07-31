/*
   FALCON - The Falcon Programming Language.
   FILE: linkerror.cpp

   Generic Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/engine.h>
#include <falcon/linkerror.h>
#include <falcon/errorclass.h>
#include <falcon/stderrors.h>

namespace Falcon {

LinkError::LinkError( const ErrorParam &params ):
   Error( Engine::instance()->stdErrors()->link(), params )
{
}

LinkError::~LinkError()
{
}

}

/* end of linkerror.cpp */
