/*
   FALCON - The Falcon Programming Language.
   FILE: matherror.cpp

   Error class raised when invalid mathematical operations are performed.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai & Paul Davey
   Begin: Sat, 07 May 2011 19:29:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/matherror.h>
#include <falcon/engine.h>
#include <falcon/stderrors.h>

namespace Falcon {

MathError::MathError( const ErrorParam &params ):
   Error( Engine::instance()->stdErrors()->math(), params )
{
}

MathError::~MathError()
{
}

}

/* end of matherror.cpp */
