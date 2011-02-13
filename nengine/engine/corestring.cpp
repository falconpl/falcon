/*
   FALCON - The Falcon Programming Language.
   FILE: corestring.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/corestring.h>
#include <falcon/itemid.h>

namespace Falcon {

CoreString::CoreString():
   Class("String", FLC_CLASS_ID_STRING )
{
}


CoreString::~CoreString()
{
}


void* CoreString::create(void* creationParams ) const
{
   cpars* cp = static_cast<cpars*>(creationParams);
   String* res = new String( cp->m_other );
   if ( cp->m_bufferize )
   {
      res->bufferize();
   }
   return res;
}


void CoreString::dispose( void* self ) const
{
   String* f = static_cast<String*>(self);
   delete f;
}


void* CoreString::clone( void* source ) const
{
   String* s = static_cast<String*>(source);
   return new String(*s);
}


void CoreString::serialize( Stream* stream, void* self ) const
{
   String* s = static_cast<String*>(self);
   s->serialize(stream);
}


void* CoreString::deserialize( Stream* stream ) const
{
   String* s = new String;
   try {
      s->deserialize( stream, false );
   }
   catch( ... )
   {
      delete s;
      throw;
   }

   return s;
}

}

/* end of corestring.cpp */
