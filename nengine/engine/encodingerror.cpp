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

#include <falcon/engine.h>
#include <falcon/encodingerror.h>
#include <falcon/errorclass.h>
#include <falcon/stderrors.h>

namespace Falcon {

EncodingError::EncodingError( const ErrorParam &params ):
   Error( Engine::instance()->stdErrors()->encoding(), params )
{
}

EncodingError::~EncodingError()
{
}

}

/* end of encodingerror.cpp */
