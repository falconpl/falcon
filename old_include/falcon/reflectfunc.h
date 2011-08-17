/*
   FALCON - The Falcon Programming Language.
   FILE: reflectionFunc.h

   Generic property reflection function definition.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 18 Jun 2008 22:33:55 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Generic property reflection function definition.
*/

#ifndef FALCON_REFLECTION_FUNC_H
#define FALCON_REFLECTION_FUNC_H

#include <falcon/setup.h>

namespace Falcon
{

class CoreObject;
class Item;

struct PropEntry;

/** Reflection type enumeration.
   Determines how reflective properties are accounted for.
*/
typedef enum {
   e_reflectNone = 0,
   e_reflectBool,
   e_reflectByte,
   e_reflectChar,
   e_reflectShort,
   e_reflectUShort,
   e_reflectInt,
   e_reflectUInt,
   e_reflectLong,
   e_reflectULong,
   e_reflectLL,
   e_reflectULL,
   e_reflectFloat,
   e_reflectDouble,
   e_reflectFunc,
   e_reflectSetGet
} t_reflection;

/** Callback function for reflective properties needing complete reflection.

   It is possible to give complete reflection to properties by assigning
   the VarDef (definition of the property type) a reflection function.

   When a configureFrom/configureTo is explicitly asked, or if the user data
   is reflective, this function may be called to configure the property
   value.

   \param instance The object were the user_data to be reflected is stored.
      The virtual machine that is involved in the operation can be retreived from there;
      errors on variable type or conditional read only settings can be enforced by raising
      on this VM.
   \param user_data The data on which reflection is going to be performed. It is not
      necessarily the user_data stored inside the instance, as the model allows to
      get or set reflective data also from/to external sources.
   \param property The property to be set or to get the setting from. When retreiving,
      the property is the original item that will be copied in the final property
      on success. When setting, it's the phisical property in the object to be
      set by this method.
   \param entry The entry descriptor indicating the property that has been called.
   \return When setting the value, should return false if the value is incompatible with
      the final object data structure (that is, if a parameter error should have been
      raised on the virtual machine).
*/

typedef void (*reflectionFunc)(CoreObject *instance, void *user_data, Item &property, const PropEntry& entry );
typedef void reflectionFuncDecl(CoreObject *instance, void *user_data, Item &property, const PropEntry& entry );

}

/** Little macro to automatize reflection of strings.
   This is not a very clean programming technique, but is effective.
   Used inside a reflectionFunc with parameters named as the standard one (i.e.
   as those provided in the function declaration), it is able to correctly
   reflect a string property into an object which provides an accessor to a Falcon::String.

   The macro raises a standard parameter error if the property is not a string during setting,
   and checks for the property being unchanged before resetting it to a new string during
   reading. As it raises a ParamError, proper .h files (falcon/error.h) must be included.

   Used mainly by the engine, users may find it useful.

   \param obj the object (pointer) containing the string to be reflected.
   \param accessor the name of the accessor used to read the variable.
*/
#define FALCON_REFLECT_STRING_FROM( obj, accessor ) \
   property = new CoreString( obj->accessor() );

/** Little macro to automatize reflection of strings.
   This stores the data coming from the engine into the object via an accessor.

   Used mainly by the engine, users may find it useful.
   \see FALCON_REFLECT_STRING_FROM
   \param obj the object (pointer) containing the string to be reflected.
   \param accessor the name of the accessor used to read the variable.
*/
#define FALCON_REFLECT_STRING_TO( obj, accessor ) \
   if ( ! property.isString() ) {\
      throw new ParamError( ErrorParam( e_inv_params ).extra( "S" ) );\
   }\
   obj->accessor( *property.asString() );\


/** Little macro to automatize reflection of integers.

   \see FALCON_REFLECT_STRING_FROM

   \param obj the object (pointer) containing the string to be reflected.
   \param accessor the name of the accessor used to read the variable.
*/
#define FALCON_REFLECT_INTEGER_FROM( obj, accessor ) \
      property = (int64) obj->accessor();\

/** Little macro to automatize reflection of integers.

   \see FALCON_REFLECT_STRING_TO

   \param obj the object (pointer) containing the string to be reflected.
   \param accessor the name of the accessor used to read the variable.
*/
#define FALCON_REFLECT_INTEGER_TO( obj, accessor ) \
   if ( ! property.isOrdinal() ) {\
      throw new ParamError( ErrorParam( e_inv_params ).extra( "N" ) );\
   }\
   obj->accessor( (uint32) property.forceInteger() );\


#endif /* FALCON_REFLECTION_FUNC_H */

/* end of reflectionFunc.h */
