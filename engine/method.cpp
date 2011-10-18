/*
   FALCON - The Falcon Programming Language.
   FILE: method.cpp

   Encapsulation for user-defined Methods in ClassUser.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 07 Aug 2011 20:18:45 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/method.cpp"

#include <falcon/method.h>
#include <falcon/item.h>

#include <falcon/classes/classuser.h>

#include <falcon/errors/codeerror.h>

namespace Falcon {

Method::Method( ClassUser* owner, const String& name, Module* mod ):
   Function( name, mod ),
   m_prop( this, owner, name )
{
   methodOf( owner );
}



Method::MethodProp::MethodProp( Method* mth, ClassUser* owner, const String& name ):
   Property( owner, name ),
   m_mth( mth )
   
{}
      
void Method::MethodProp::set( void*, const Item& )
{
   throw new CodeError( ErrorParam( e_prop_ro, __LINE__, SRC )
      .extra( name() )
      );
}

void Method::MethodProp::get( void* instance, Item& target )
{
   target.setUser( owner(), instance );
   target.methodize( m_mth );
}   

}

/* end of method.cpp */
