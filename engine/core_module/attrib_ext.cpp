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
