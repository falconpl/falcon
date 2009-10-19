/*
   FALCON - The Falcon Programming Language.
   FILE: attrib_ext.cpp

   Facilities handling attributes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jul 2009 23:26:40 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/attribmap.h>
#include <falcon/corefunc.h>
#include "core_module.h"

namespace Falcon {
namespace core {

static void inner_make_item( VarDef* vd, Item& itm )
{
   switch( vd->type() )
   {
      case VarDef::t_bool: itm.setBoolean( vd->asBool() ); break;
      case VarDef::t_int: itm.setInteger( vd->asInteger() ); break;
      case VarDef::t_num: itm.setNumeric( vd->asNumeric() ); break;
      case VarDef::t_string:
      {
         itm.setString( new CoreString( *vd->asString() ) );
      }
      break;

      default:
         itm.setNil();
   }
}

static CoreDict* interal_make_attrib_dict( Map* attr )
{
   CoreDict* cd = new CoreDict( new LinearDict( attr->size() ) );

   MapIterator iter = attr->begin();
   while( iter.hasCurrent() )
   {
      VarDef* vd = *(VarDef**) iter.currentValue();
      Item temp;
      inner_make_item( vd, temp );

      cd->put( new CoreString(
         *(String*) iter.currentKey() ),
         temp
         );
      iter.next();
   }

   return cd;
}


/*#
   @function attributes
   @brief Returns a dictionary containing annotation attributes of the current module.
   @return Nil if the current module has no attributes, or a string-indexed dictionary.

   @see Function.attributes
   @see Class.attributes
   @see Object.attributes
*/

FALCON_FUNC  attributes ( ::Falcon::VMachine *vm )
{
   // we want to know the attributes of the module calling us.
   StackFrame* cf = vm->currentFrame();
   const Module* mod = cf->m_module->module();

   Map* attr =  mod->attributes();
   if( attr != 0 )
   {
      vm->retval( interal_make_attrib_dict( attr ) );
   }
}

/*#
   @method attributes Class
   @brief Returns a dictionary containing annotation attributes of the given class.
   @return Nil if the class has no attributes, or a string-indexed dictionary.

   @see attributes
*/

FALCON_FUNC  Class_attributes ( ::Falcon::VMachine *vm )
{
   Map* attr = vm->self().asClass()->symbol()->getClassDef()->attributes();

   if( attr != 0 )
   {
      vm->retval( interal_make_attrib_dict( attr ) );
   }
}

/*#
   @method attributes Object
   @brief Returns a dictionary containing annotation attributes of the given object.
   @return Nil if the object has no attributes, or a string-indexed dictionary.

   If the object is a class instance, this method will return the attributes of
   the generator class.

   @see attributes
*/

FALCON_FUNC  Object_attributes ( ::Falcon::VMachine *vm )
{
   Map* attr = vm->self().asObject()->generator()->symbol()->getClassDef()->attributes();

   if( attr != 0 )
   {
      vm->retval( interal_make_attrib_dict( attr ) );
   }
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
      throw new AccessError( ErrorParam( e_inv_params, __LINE__ )
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

/*#
   @method attributes Function
   @brief Returns a dictionary containing annotation attributes of the given function.
   @return Nil if the function has no attributes, or a string-indexed dictionary.

   @see attributes
*/

FALCON_FUNC  Function_attributes ( ::Falcon::VMachine *vm )
{
   const Symbol* sym = vm->self().asFunction()->symbol();

   // currently, extfunc are not supported; let the VM return nil
   if ( sym->isExtFunc() )
      return;

   Map* attr = sym->getFuncDef()->attributes();

   if( attr != 0 )
   {
      vm->retval( interal_make_attrib_dict( attr ) );
   }
}

/*#
   @method attributes Method
   @brief Returns the attributes associated with the method function.
   @return Nil if the function has no attributes, or a string-indexed dictionary.

   @see attributes
*/

/*#
   @method attributes ClassMethod
   @brief Returns the attributes associated with the method function.
   @return Nil if the function has no attributes, or a string-indexed dictionary.

   @see attributes
*/

FALCON_FUNC  Method_attributes ( ::Falcon::VMachine *vm )
{
   if ( ! vm->self().asMethodFunc()->isFunc() )
      return;

   const Symbol* sym = static_cast<CoreFunc*>(vm->self().asMethodFunc())->symbol();

   // currently, extfunc are not supported; let the VM return nil
   if ( sym->isExtFunc() )
      return;

   Map* attr = sym->getFuncDef()->attributes();
   if( attr != 0 )
   {
      vm->retval( interal_make_attrib_dict( attr ) );
   }
}

}
}

/* end of attrib_ext.cpp */
