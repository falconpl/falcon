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
   Interface extension functions
*/

#include <falcon/engine.h>
#include "MP_mod.h"
#include "MP_ext.h"
#include "MP_st.h"

namespace Falcon {
namespace Ext {

// The following is a faldoc block for the function
/*--#  << change this to activate.
   @function skeleton
   @brief A basic script function.
   @return Zero.

   This function just illustrates how to bind the ineer MOD logic
   with the script. Also, Mod::skeleton(), used by this
   function, is exported through the "service", so it is
   possible to call the MOD logic directly from an embedding
   application, once loaded the module and accessed the service.
*/

FALCON_FUNC  MPZ_init( ::Falcon::VMachine *vm )
{
  Item *i_value = vm->param(0);
  Item *i_self  = &vm->self();

  if ( !i_self->isOfClass( "MPZ" ) )
  {
    throw new Falcon::TypeError( 
			Falcon::ErrorParam( Falcon::e_not_implemented, __LINE__ )
        .extra( "This should never ever ever happen!" ) );
  }

  if ( i_value == 0 )
  {
    i_self->asObject()->setUserData(new Mod::MPZ_carrier());
  }
  else if  ( i_value->isInteger() )
  {
    i_self->asObject()->setUserData(new Mod::MPZ_carrier(i_value->asInteger()));
  }
  else if ( i_value->isNumeric() )
  {
    i_self->asObject()->setUserData(new Mod::MPZ_carrier(i_value->asNumeric()));
  }
  else if ( i_value->isString() )
  {
    Item *i_base = vm->param(1);
    int64 base;
    if ( i_base != 0 && ( i_base->isNumeric() || i_base->isInteger() ) )
    {
      base = i_base->forceInteger();
    }
    else
    {
      base = 0;
    }
    i_self->asObject()->setUserData(new Mod::MPZ_carrier(i_value->asString(), base));
  }
  else
  {
    throw new Falcon::ParamError( 
			Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
        .extra( "N or S,[N]" ) );
  }


}

/*--#  << change this to activate.
   @function __add
   @brief Addition of MP Integers.
   @return The sum of the MPZ and the other parameter.
  
   Addition of Multi-Precision Integers.  
   Supports MPInts, standard integers as well as floating point numbers MPRationals and MPFloats.
*/
FALCON_FUNC  MPZ_add( ::Falcon::VMachine *vm )
{
  Item *i_other = vm->param(0);
  if ( i_other == 0 )
  {
    throw new Falcon::ParamError( 
			Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
        .extra( "N or MPZ or MPQ or MPF" ) );
  }
  Item *i_inplace = vm->param(1);
  Item *i_self  = &vm->self();
  CoreClass *mpz_class = vm->findWKI("MPZ")->asClass();
  CoreObject *result;
  if ( i_inplace == 0 || !i_inplace->isTrue() )
  {
    result = mpz_class->createInstance();
    Mod::MPZ_carrier *result_data = new Mod::MPZ_carrier();
    result->setUserData(result_data);
  }
  else
  {
    result = i_self->asObject();
  }

  if ( i_other->isInteger() )
  {
    Mod::MPZ_carrier otherMPZ(i_other->asInteger());
  }
  else if ( i_other->isNumeric() )
  {
    Mod::MPZ_carrier otherMPZ(i_other->asNumeric());
  } 
  else if ( i_other->isOfClass("MPZ") )
  {
    Mod::MPZ_carrier *otherMPZ = static_cast<Mod::MPZ_carrier *>(i_other->asObject()->getFalconData());
  }
  else 
  {
    throw new Falcon::ParamError( 
			Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
        .extra( "N or MPZ or MPQ or MPF" ) );
  }

  vm->retval(result);
}

/*--#  << change this to activate.
   @function __sub
   @brief Subtraction of MP Integers.
   @return The difference between the MPZ and the other parameter.
  
   Subtraction of Multi-Precision Integers.  
   Supports MPInts, standard integers as well as floating point numbers MPRationals and MPFloats.
*/
FALCON_FUNC  MPZ_sub( ::Falcon::VMachine *vm )
{
  Item *i_other = vm->param(0);
  if ( i_other == 0 )
  {
    throw new Falcon::ParamError( 
			Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
        .extra( "N or MPZ or MPQ or MPF" ) );
  }
  Item *i_inplace = vm->param(1);
  Item *i_self  = &vm->self();
  CoreClass *mpz_class = vm->findWKI("MPZ")->asClass();
  CoreObject *result;
  if ( i_inplace == 0 || !i_inplace->isTrue() )
  {
    result = mpz_class->createInstance();
    Mod::MPZ_carrier *result_data = new Mod::MPZ_carrier();
    result->setUserData(result_data);
  }
  else
  {
    result = i_self->asObject();
  }

  if ( i_other->isInteger() )
  {
    Mod::MPZ_carrier otherMPZ(i_other->asInteger());
  }
  else if ( i_other->isNumeric() )
  {
    Mod::MPZ_carrier otherMPZ(i_other->asNumeric());
  } 
  else if ( i_other->isOfClass("MPZ") )
  {
    Mod::MPZ_carrier *otherMPZ = static_cast<Mod::MPZ_carrier *>(i_other->asObject()->getFalconData());
  }
  else 
  {
    throw new Falcon::ParamError( 
			Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
        .extra( "N or MPZ or MPQ or MPF" ) );
  }

  vm->retval(result);
}

FALCON_FUNC  MPZ_toString( ::Falcon::VMachine *vm )
{
  Item *i_base = vm->param(0);
  CoreObject *self = vm->self().asObject();
  int64 base;
  if ( i_base != 0 && ( i_base->isNumeric() || i_base->isInteger() ) )
  {
    base = i_base->forceInteger();
  }
  else
  {
    base = 10;
  }
  vm->retval(static_cast<Mod::MPZ_carrier *>(self->getFalconData())->toString(base));

}

}
}

/* end of MP_mod.cpp */
