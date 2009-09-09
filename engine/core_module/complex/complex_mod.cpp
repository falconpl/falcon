/*
   FALCON - The Falcon Programming Language.
   FILE: complex_ext.cpp

   Complex class for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Enrico Lumetti
   Begin: Sat, 05 Sep 2009 21:04:31 +0000

   -------------------------------------------------------------------
   (C) Copyright 2009: The above AUTHOR

         Licensed under the Falcon Programming Language License,
      Version 1.1 (the "License"); you may not use this file
      except in compliance with the License. You may obtain
      a copy of the License at

         http://www.falconpl.org/?page_id=license_1_1

      Unless required by applicable law or agreed to in writing,
      software distributed under the License is distributed on
      an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
      KIND, either express or implied. See the License for the
      specific language governing permissions and limitations
      under the License.

*/

/** \file
   Complex class for Falcon
   Internal logic functions - implementation.
*/

#include "complex_mod.h"

namespace Falcon {
namespace Mod {


CoreObject *Complex_Factory( const CoreClass *cls, void *, bool )
{
    return new Complex ( cls );
}

Complex ::~Complex ( void )
{ }

CoreObject* Complex::clone( void ) const
{
   return new Complex( *this );
}

bool Complex ::hasProperty( const String &key ) const
{
    uint32 pos = 0;
    return generator()->properties().findKey( key, pos );
}


bool Complex ::setProperty( const String &key, const Item &ret )
{
    if (key == "real") 
    {
	return true;
    }
    if (key == "imag") 
    {
	return true;
    }
    return false;
}

bool Complex ::getProperty( const String &key, Item &ret ) const
{
    if (key == "real")
    {
    	ret.setNumeric(m_real);
    	return true;
    }
    if (key == "imag")
    {
    	ret.setNumeric(m_imag);
    	return true;
    }
    return defaultProperty( key, ret );
}


}
}


/* end of complex_mod.cpp */
