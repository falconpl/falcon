/*
   FALCON - The Falcon Programming Language.
   FILE: MP_ext.cpp

   Multi-Precision Math support
   Interface extension functions
   -------------------------------------------------------------------
   Author: Paul Davey
   Begin: Fri, 12 Mar 2010 15:58:42 +0000

   -------------------------------------------------------------------
   (C) Copyright 2010: The above AUTHOR

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
   Multi-Precision Math support
   Internal logic functions - declarations.
*/

#ifndef MP_mod_H
#define MP_mod_H


#include <falcon/falcondata.h>
#include <falcon/string.h>


namespace Falcon {
namespace Mod {

  class MPZ_carrier : public FalconData
  {
  public:
	MPZ_carrier();
	MPZ_carrier( int64 num );
    MPZ_carrier( double num );
    MPZ_carrier( String *num, int64 base );
    MPZ_carrier( const MPZ_carrier &otherMPZ );
    
	~MPZ_carrier();

    virtual void gcMark( uint32 mark ){}
    virtual MPZ_carrier *clone() const;

    virtual bool serialize( Stream *stream, bool bLive ) const;
	virtual bool deserialize( Stream *stream, bool bLive );

    String *toString( int64 base );
  };

}
}

#endif

/* end of MP_mod.h */
