/*
   FALCON - The Falcon Programming Language.
   FILE: codeerror.cpp

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/errors/codeerror.h>
#include <falcon/classes/classerror.h>
#include <falcon/engine.h>
#include <falcon/stderrors.h>

namespace Falcon {

CodeError::CodeError( const ErrorParam &params ):
   Error( Engine::instance()->stdErrors()->code(), params )
{
}

CodeError::~CodeError()
{
}

}

/* end of codeerror.cpp */
