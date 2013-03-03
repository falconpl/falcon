/*
   FALCON - The Falcon Programming Language.
   FILE: unserializableerror.cpp

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/errors/unserializableerror.h>
#include <falcon/engine.h>
#include <falcon/stderrors.h>

namespace Falcon {

UnserializableError::UnserializableError( ):
   Error( Engine::instance()->stdErrors()->unserializable() )
{
}

UnserializableError::UnserializableError( const ErrorParam &params ):
   Error( Engine::instance()->stdErrors()->unserializable(), params )
{
}

UnserializableError::~UnserializableError()
{
}

}

/* end of unserializableerror.cpp */
