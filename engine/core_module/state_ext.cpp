/*
   FALCON - The Falcon Programming Language.
   FILE: attrib_ext.cpp

   Facilities handling attributes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 15 Nov 2009 11:17:19 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/corefunc.h>
#include "core_module.h"

namespace Falcon {
namespace core {


/*#
   @method setState Object
   @param nstate The new state into which the object is moved.
   @brief Change the current active state of an object.
   @return Return value of the __leave -> __enter sequence, if any, or nil
   @raise CodeError if the state is not part of the object state.

   This method changes the state of the object, applying a new set of function
   described in the state section.
*/
FALCON_FUNC  Object_setState ( ::Falcon::VMachine *vm )
{
   Item* nstate = vm->param(0);
   if ( nstate == 0 || ! nstate->isString() )
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
            .extra( "S") );

   // will raise in case of problems.
   vm->self().asObject()->setState( *nstate->asString(), vm );
}


/*#
   @method getState Object
   @brief Return the current state of an object.
   @return A string representing the current state of an object, or nil if the object is stateless.

   This function returns the current state in which an object is operating.

*/

FALCON_FUNC  Object_getState( ::Falcon::VMachine *vm )
{
   CoreObject* obj = vm->self().asObject();
   if( obj->hasState() )
      vm->retval( new CoreString( obj->state() ) );
   else
      vm->retnil();
}

/*#
   @method apply Object
   @brief Applies the values in a dictionary to the corresponding properties.
   @param dict A "stamp" dictionary.
   @raise AccessError if some property listed in the dictionary is not defined.
   @return This same object.

   This method applies a "stamp" on this object. The idea is that of copying
   the contents of all the items in the dictionary into the properties of this
   object. Dictionaries are more flexible than objects, at times they are preferred
   for i.e. network operations and key/value databases. With this method, you
   can transfer data from a dictionary in an object with a single VM step, paying
   just the cost of the copy; in other words, sparing the VM operations needed
   for looping over the dictionary and searching dynamically the required properties.

   @note Non-string keys in @b dict are simply skipped.

   @see Object.retrieve
*/

FALCON_FUNC  Object_apply( ::Falcon::VMachine *vm )
{
   Item* i_dict = vm->param( 0 );

   if ( i_dict == 0 || ! i_dict->isDict() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
            .extra( "D") );
   }

   CoreObject* self = vm->self().asObject();
   self->apply( i_dict->asDict()->items(), true );
   vm->retval( self );
}

/*#
   @method retrieve Object
   @brief Gets the values stored in the properties of this object.
   @optparam dict A "stamp" dictionary.
   @raise AccessError if some property listed in the dictionary is not defined.
   @return A dictionary containing the contents of each property (stored as a key
         in the dictionary).

   This method takes all the data values stored in the properties of this object
   (ignoring methods), and places them in a dictionary. Property names are used
   as keys under which to store flat copies of the property values.

   If a @b dict parameter is passed, this method will take only the properties
   stored as keys, and eventually raise an AccessError if some of them are not found.

   Otherwise, a new dictionary will be filled with all the properties in this object.

   @note In case of repeated activity, the same dictionary can be used to fetch
   new values to spare memory and CPU.

   @see Object.apply
*/

FALCON_FUNC  Object_retrieve( ::Falcon::VMachine *vm )
{
   Item* i_dict = vm->param( 0 );
   CoreDict* dict;
   bool bFillDict;

   if( i_dict == 0 )
   {
      bFillDict = true;
      dict = new CoreDict( new LinearDict );
   }
   else
   {
      if ( ! i_dict->isDict() )
      {
         throw new AccessError( ErrorParam( e_inv_params, __LINE__ )
               .origin(e_orig_runtime)
               .extra( "[D]" ) );
      }

      dict = i_dict->asDict();
      bFillDict = false;
   }

   CoreObject* self = vm->self().asObject();
   self->retrieve( dict->items(), true, bFillDict, true );
   vm->retval( dict );
}

}
}

