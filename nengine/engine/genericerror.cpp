/*
   FALCON - The Falcon Programming Language.
   FILE: genericerror.h

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/genericerror.h>
#include <falcon/errorclass.h>

namespace Falcon {

class GenericErrorClass: public ErrorClass
{
public:
   GenericErrorClass():
      ErrorClass( "GenericError" )
      {}
   
   virtual void* create(void* creationParams ) const
   {
      return new GenericError( *static_cast<ErrorParam*>(creationParams) );
   }
};

static GenericErrorClass s_ec;

GenericError::GenericError( const ErrorParam &params ):
   Error( &s_ec, params )
{
}

GenericError::~GenericError()
{
}

}

/* end of codeerror.h */
