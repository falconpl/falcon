/*
   FALCON - The Falcon Programming Language.
   FILE: syntaxerror.cpp

   Error class representing compilation errors.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Apr 2011 20:13:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/syntaxerror.h>
#include <falcon/engine.h>
#include <falcon/stderrors.h>

namespace Falcon {

SyntaxError::SyntaxError( const ErrorParam &params ):
   Error( Engine::instance()->stdErrors()->syntax(), params )
{
}

SyntaxError::~SyntaxError()
{
}

}

/* end of syntaxerror.cpp */
