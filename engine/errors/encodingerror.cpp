/*
   FALCON - The Falcon Programming Language.
   FILE: encodingerror.cpp

   Encoding Error Class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Mar 2011 13:04:14 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/classes/classerror.h>

#include <falcon/engine.h>
#include <falcon/errors/encodingerror.h>
#include <falcon/stderrors.h>

namespace Falcon {

EncodingError::EncodingError( ):
   Error( Engine::instance()->stdErrors()->encoding() )
{
}

EncodingError::EncodingError( const ErrorParam &params ):
   Error( Engine::instance()->stdErrors()->encoding(), params )
{
}

EncodingError::~EncodingError()
{
}

}

/* end of encodingerror.cpp */
