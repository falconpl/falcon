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
#include <falcon/errors/codeerror.h>

#include <map>

namespace Falcon {

class BOM::Private
{
public:
   typedef std::map<String, BOM::handler> HandlerMap;

   HandlerMap m_handlers;
};

BOM::BOM():
   _p( new Private )
{
   Private::HandlerMap& hm = _p->m_handlers;

   hm["len"] = BOMH::len;

   hm["baseClass"] = &BOMH::baseClass;
   hm["bound"] = BOMH::bound;
   hm["className"] = BOMH::className;
   hm["clone"] = BOMH::clone;
   hm["describe"] = BOMH::describe;
   hm["isCallable"] = BOMH::isCallable;
   hm["ptr"] = BOMH::ptr;
   hm["toString"] = BOMH::toString;
   hm["typeId"] = BOMH::typeId;
   
   hm["compare"] = BOMH::compare;
   hm["derivedFrom"] = BOMH::derivedFrom;

   hm["get"] = BOMH::get;
   hm["set"] = BOMH::set;
   hm["has"] = BOMH::has;
   hm["properties"] = BOMH::properties;
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


void bound(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );
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


void ptr(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );
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

void get(VMContext* ctx, const Class*, void*)
{
  static Function* func = static_cast<Function*>(Engine::instance()->getMantra("get"));
  fassert( func != 0 );

  Item &value = ctx->topData();
  value.methodize(func);
}

void set(VMContext* ctx, const Class*, void*)
{
  static Function* func = static_cast<Function*>(Engine::instance()->getMantra("set"));
  fassert( func != 0 );

  Item &value = ctx->topData();
  value.methodize(func);
}

void has(VMContext* ctx, const Class*, void*)
{
  static Function* func = static_cast<Function*>(Engine::instance()->getMantra("has"));
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

}

}

/* end of bom.cpp */
