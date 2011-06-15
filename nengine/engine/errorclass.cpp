/*
   FALCON - The Falcon Programming Language.
   FILE: errorclass.cpp

   Class for storing error in scripts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/errorclass.h>

namespace Falcon {

ErrorClass::ErrorClass( const String& name ):
   Class(name)
{
}

ErrorClass::~ErrorClass()
{
}

void ErrorClass::dispose( void* self ) const
{
   // use the virtual delete feature.
   Error* error = (Error*)self;
   error->decref();
}

void* ErrorClass::clone( void* ) const
{
   // errors are uncloneable for now
   return false;
}

void ErrorClass::serialize( DataWriter*, void* ) const
{
   // TODO
}

void* ErrorClass::deserialize( DataReader* ) const
{
   //TODO
   return 0;
}

}

/* end of errorclass.cpp */

