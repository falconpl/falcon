/*
   FALCON - The Falcon Programming Language.
   FILE: bom.cpp

   Basic object model support.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 18:17:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/bom.cpp"

#include <falcon/bom.h>
#include <falcon/string.h>
#include <falcon/vm.h>
#include <falcon/optoken.h>
#include <falcon/class.h>
#include <falcon/pseudofunc.h>
#include <falcon/module.h>
#include <falcon/class.h>
#include <falcon/stderrors.h>

#include <map>

namespace Falcon {

class BOM::Private
{
public:
   typedef std::map<String, BOM::handler> HandlerMap;

   HandlerMap m_handlers;
};

/*#
  @class BOM
  @brief Basic object model

  @prop len  Generic length of a sequence, or size of an object.
  @prop baseClass Class of which the given entity is an instance of.
  @prop className Name of the base class (as a string).
  @prop isCallable True if the item is a directly callable entity (function, method or code).
  @prop typeId Numeric type id associated with the item class.


 */

/*#
 @method clone BOM
 @brief clone an item
 @raise UncloneableError if the class of this item doesn't support cloning.
 */

/*#
 @method describe BOM
 @brief provides a readable description or in-depth of the item.

 Differs from @a BOM.toString as @b describe is meant to be used during development
 to inspect the internals of a given entity, or some meta-information about
 its status, while @b toString is meant to return a end-user representation of the
 entity.
 */

/*#
 @method compare BOM
 @brief Provides a basic comparison criterion used for default ordering.
 @param other The other object to be compared with.
 @return < 0 if this item is smaller than @b other, > 0 if it's greather or 0 if they are the same.

 */

/*#
 @method derivedFrom BOM
 @brief True if this entity is an instance of the given class, or of a child of the given class.
 @param cls The class.
 @return true if this is an instance, a subclass or a subclass instance of @b cls.
 */

/*#
 @method getp BOM
 @brief gets a property value by name.
 @param prop The property name.
 @return The value of the given property.
 @raise AccessError if the property doesn't exist.

 */

/*#
 @method setp BOM
 @brief sets a property value by name.
 @param prop The property name.
 @param value The value to be set
 @raise AccessError if the property doesn't exist, and the target entity doesn't support dynamic properties.
 */

/*#
 @method hasp BOM
 @brief checks if a given property exists.
 @param prop The property name.
 @return true if the property exists, false if not.
 */

/*#
 @method approp BOM
 @brief Applies a set of values to the properties of an item.
 @brief item Item on which the properties are to be applied.
 @param props Properties to be applied, in a dictionary of [property_name => value].
 @return The @b item itself.

 This function/method applies the values in the dictionary passed as @props to the
 properties that are found in the target item.

 For instance, the following:
 @code
   something.approp( ["a"=> 1, "b" => 2] )
 @endcode

 sets the value 1 in the property @b a of the item @b something, and 2 in the @b b property.
 */

/*#
 @property properties BOM
 @brief an array containing all the property names exposed publicly by this entity.

 This property contains the dynamic property only. In Falcon class instances, statically
 declared methods won't be returned.

 @note Each access to this property creates a new instance of the array. For repeated use,
 cache the array in a local variable.
 */

/*#
 @method render BOM
 @brief generates a Falcon source code representation of the entity.
 @optparam stream A stream on which to store the rendering.

 This method creates a source-code representation of the given object. This
 is particularly useful with entities representing Falcon source code snippets,
 classes or prototypes.

 Otherwise, it defaults to @b describe method.
 */

/*#
 @method foreach BOM
 @brief Iterates over all the contents of a sequence.
 @param code A code to be evaluated with each entry of the sequence.

 This method uses the iter/next operators provided by the class and feeds the
 result in the given code (that can be any evaluable entity). This is equivalent
 to having the entity set as the iterable entity in a for/in statement.

 In other words this is the functional version of the for/in statement.

 @note Integer number can be iterated as well, and will go 0 to the given number.
 For instance, 5.foreach(printl) will print all the numbers from 0 to 5.
 */

/*#
 @method summon BOM
 @brief Indirectly performs a summoning by name.
 @param msg The summoning message.
 @optparam ... The parameters for the summoning.
 @return The summon result.
 */

/*#
 @method vsummon BOM
 @brief Indirectly performs a summoning by name.
 @param msg The summoning message.
 @param params An array containing the parameters to be passed to the summoning.
 @return The summon result.
 */

/*#
 @method delegate BOM
 @brief Delegates one or more messages handling to the given entity.
 @optparam target The delegate.
 @optparam ... The messages to be delegated, by name.

 Call without parameters to un-delegate any previously delegated message. Use @b nil
 as target to un-delegate the specified messages.
 */

/*#
 @method respondsTo BOM
 @brief Checks if the entity wants to respond to this summoning.
 @param msg The summoning message.
 @return true if the message is handled by this entity.
 */


BOM::BOM():
   _p( new Private )
{
   Private::HandlerMap& hm = _p->m_handlers;

   hm["len"] = &BOMH::len;

   hm["baseClass"] = &BOMH::baseClass;
   hm["className"] = &BOMH::className;
   hm["clone"] = &BOMH::clone;
   hm["describe"] = &BOMH::describe;
   hm["isCallable"] = &BOMH::isCallable;
   hm["toString"] = &BOMH::toString;
   hm["typeId"] = &BOMH::typeId;
   
   hm["compare"] = &BOMH::compare;
   hm["derivedFrom"] = &BOMH::derivedFrom;

   hm["getp"] = &BOMH::getp;
   hm["setp"] = &BOMH::setp;
   hm["hasp"] = &BOMH::hasp;
   hm["properties"] = &BOMH::properties;
   hm["approp"] = &BOMH::approp;

   hm["render"] = &BOMH::render;
   hm["foreach"] = &BOMH::foreach;
}

BOM::~BOM()
{
   delete _p;
}

BOM::handler BOM::get( const String& property ) const
{
   Private::HandlerMap& hm = _p->m_handlers;
   Private::HandlerMap::iterator iter = hm.find( property );

   if( iter == hm.end() )
   {
      return 0;
   }

   return iter->second;
}


namespace BOMH
{

//==================================================
// Len method.
//

void len(VMContext* ctx, const Class*, void*)
{
   Item& topData = ctx->topData();
   topData = topData.len();
}


void className(VMContext* ctx, const Class*, void* data)
{
   Item* pself;
   OpToken token( ctx, pself );

   // which means using force class inst.
   Class* cls1;
   pself->forceClassInst( cls1, data );
   token.exit(cls1->name()); // garbage this string
}

void baseClass(VMContext* ctx, const Class* cls, void*)
{
   if( ! cls->isMetaClass() )
   {
      ctx->topData().setUser( cls->handler(), const_cast<Class*>(cls) );
   }
   // otherwise the topdata is already a class
}

//======================================================
// Clone

void clone(VMContext *ctx, const Class*, void*)
{
   static Function* cloneFunc = static_cast<Function*>(Engine::instance()->getMantra("clone"));
   fassert( cloneFunc != 0 );

   Item &value = ctx->topData();
   value.methodize(cloneFunc);
}


void describe(VMContext* ctx, const Class* cls, void* data)
{
   String* target = new String;
   cls->describe( data, *target, 3, -1 );

   Item& topData = ctx->topData();
   topData.setUser( FALCON_GC_HANDLE(target) );
}


void isCallable(VMContext* ctx, const Class* cls, void* )
{
   ctx->topData().setBoolean(
       cls->typeID() == FLC_CLASS_ID_FUNC
       || cls->typeID() == FLC_CLASS_ID_TREESTEP
       || cls->typeID() == FLC_ITEM_METHOD );
}



void toString(VMContext* ctx, const Class* cls, void* data)
{
   cls->op_toString( ctx, data );
}


void typeId(VMContext* ctx, const Class*, void* data)
{
   Item* pself;
   OpToken token( ctx, pself );

   Class* cls1;
   if ( pself->asClassInst( cls1, data) )
   {
      token.exit( cls1->typeID() );
   }
   else
   {
      token.exit( pself->type() );
   }
}


void compare(VMContext* ctx, const Class*, void*)
{
   static Function* func = static_cast<Function*>(Engine::instance()->getMantra("compare"));
   fassert( func != 0 );

   Item &value = ctx->topData();
   value.methodize(func);
}


void derivedFrom(VMContext* ctx, const Class*, void*)
{
  static Function* func = static_cast<Function*>(Engine::instance()->getMantra("derivedFrom"));
  fassert( func != 0 );

  Item &value = ctx->topData();
  value.methodize(func);
}

void getp(VMContext* ctx, const Class*, void*)
{
  static Function* func = static_cast<Function*>(Engine::instance()->getMantra("getp"));
  fassert( func != 0 );

  Item &value = ctx->topData();
  value.methodize(func);
}

void setp(VMContext* ctx, const Class*, void*)
{
  static Function* func = static_cast<Function*>(Engine::instance()->getMantra("setp"));
  fassert( func != 0 );

  Item &value = ctx->topData();
  value.methodize(func);
}

void hasp(VMContext* ctx, const Class*, void*)
{
  static Function* func = static_cast<Function*>(Engine::instance()->getMantra("hasp"));
  fassert( func != 0 );

  Item &value = ctx->topData();
  value.methodize(func);
}

void properties(VMContext* ctx, const Class*, void*)
{
  static Function* func = static_cast<Function*>(Engine::instance()->getMantra("properties"));
  fassert( func != 0 );

  Item &value = ctx->topData();
  value.methodize(func);
}

void render(VMContext* ctx, const Class*, void*)
{
  static Function* func = static_cast<Function*>(Engine::instance()->getMantra("render"));
  fassert( func != 0 );

  Item &value = ctx->topData();
  value.methodize(func);
}

void foreach(VMContext* ctx, const Class*, void*)
{
  static Function* func = static_cast<Function*>(Engine::instance()->getMantra("foreach"));
  fassert( func != 0 );

  Item &value = ctx->topData();
  value.methodize(func);
}


void approp(VMContext* ctx, const Class*, void*)
{
  static Function* func = static_cast<Function*>(Engine::instance()->getMantra("approp"));
  fassert( func != 0 );

  Item &value = ctx->topData();
  value.methodize(func);
}



}

}

/* end of bom.cpp */
