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
   Interface extension functions
*/

#include <falcon/engine.h>
#include "complex_mod.h"
#include "complex_ext.h"
#include "complex_st.h"

namespace Falcon {
namespace Ext {


FALCON_FUNC Complex_init( ::Falcon::VMachine *vm )
{
    Item *i_real = vm->param(0);
    Item *i_imag = vm->param(1);

    if ( 
	 ( i_real == 0 or !( i_real->isNumeric() ) or !( i_real->isOrdinal() ) ) or
         ( i_imag == 0 or !( i_imag->isNumeric() ) or !( i_real->isOrdinal() ) )
       )
    {
	throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
	    extra( "[N,N]" ) 
	);
    }

    Mod::Complex *c = dyncast<Mod::Complex *> ( vm->self().asObject() );
    c->m_real =  i_real->forceNumeric();
    c->m_imag =  i_imag->forceNumeric();
}

FALCON_FUNC Complex_toString( ::Falcon::VMachine *vm )
{
    Mod::Complex *self = dyncast<Mod::Complex *>( vm->self().asObject() );
    String res,real,imag;
    Item(self->m_real).toString(real);
    Item(self->m_imag).toString(imag);
    res=real+","+imag+"i";
    vm->retval(res);
}

/* |(a,b)| = sqrt(a^2+b^2) */

FALCON_FUNC Complex_abs( ::Falcon::VMachine *vm )
{
    Mod::Complex *self = dyncast<Mod::Complex *>( vm->self().asObject() );
    vm->retval( sqrt( pow( self->m_real, 2)+pow( self->m_imag, 2) ) );
}

/* (a,b) + (d,c) = (a+d,b+c) */

FALCON_FUNC Complex_add__( ::Falcon::VMachine *vm )
{
    Item *i_obj = vm->param( 0 );
    
    if ( i_obj == 0 or !( i_obj->isOfClass( "Complex" ) ) )
    {
	throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
	    extra( "[Complex]" ) 
	);
    }

    Mod::Complex *ret  = dyncast<Mod::Complex *>( dyncast<Mod::Complex *>( vm->self().asObject() )->clone() );
    
    if ( i_obj->isOfClass( "Complex" ) )
    {
	Mod::Complex *obj = dyncast<Mod::Complex *>( i_obj->asObject() );
	ret->m_real+=obj->m_real;
	ret->m_imag+=obj->m_imag;
    }
    vm->retval( ret );
}

/* (a,b) * (c,d) = (ac−bd,bc+ad) */

FALCON_FUNC Complex_mul__( ::Falcon::VMachine *vm )
{
    Item *i_obj = vm->param( 0 );
    
    if ( i_obj == 0 or !( i_obj->isOfClass( "Complex" ) ) )
    {
	throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
	    extra( "[Complex]" )
	);
    }

    Mod::Complex *ret  = dyncast<Mod::Complex *>( dyncast<Mod::Complex *>( vm->self().asObject() )->clone() );
    
    if ( i_obj->isOfClass( "Complex" ) )
    {
	Mod::Complex *obj = dyncast<Mod::Complex *>( i_obj->asObject() );
	ret->m_real = ret->m_real*obj->m_real - ret->m_imag*obj->m_imag;
	ret->m_imag = ret->m_imag*obj->m_real + ret->m_real*obj->m_imag;
    }
    vm->retval( ret );
}



}
}

/* end of complex_mod.cpp */
