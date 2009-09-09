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
   Internal logic functions - declarations.
*/

#ifndef complex_mod_H
#define complex_mod_H

#include <falcon/cclass.h>
#include <falcon/reflectobject.h>
#include <falcon/memory.h>
#include <math.h>

namespace Falcon {
namespace Mod {

CoreObject *Complex_Factory( const CoreClass *cls, void *, bool );

class Complex : public CoreObject
{
    public:
	double m_real;
	double m_imag;

	Complex ( const CoreClass *cls ):
	    CoreObject( cls )
	{ }

	virtual ~Complex ( void );
	virtual void gcMark( uint32 ) { }
	virtual CoreObject *clone( void ) const;

	virtual bool hasProperty( const String &key ) const;
	virtual bool setProperty( const String &key, const Item &ret );
	virtual bool getProperty( const String &key, Item &ret ) const;
};

}
}

#endif

/* end of complex_mod.h */
