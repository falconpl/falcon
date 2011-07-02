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

#include <falcon/bom.h>
#include <falcon/string.h>
#include <falcon/vm.h>
#include <falcon/optoken.h>
#include <falcon/class.h>
#include <falcon/pseudofunc.h>
#include <falcon/module.h>
#include <falcon/class.h>

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
   hm["len_"] = BOMH::len_;

   hm["baseClass"] = &BOMH::baseClass;
   hm["baseClass_"] = BOMH::baseClass_;
   hm["bound"] = BOMH::bound;
   hm["bound_"] = BOMH::bound_;
   hm["className"] = BOMH::className;
   hm["className_"] = BOMH::className_;
   hm["clone"] = BOMH::clone;
   hm["clone_"] = BOMH::clone_;
   hm["describe"] = BOMH::describe;
   hm["describe_"] = BOMH::describe_;
   hm["isCallable"] = BOMH::isCallable;
   hm["isCallable_"] = BOMH::isCallable_;
   hm["metaclass"] = BOMH::metaclass;
   hm["metaclass_"] = BOMH::metaclass_;
   hm["ptr"] = BOMH::ptr;
   hm["ptr_"] = BOMH::ptr_;
   hm["toString"] = BOMH::toString;
   hm["toString_"] = BOMH::toString_;
   hm["typeId"] = BOMH::typeId;
   hm["typeId_"] = BOMH::typeId_;
   
   hm["compare"] = BOMH::compare;
   hm["baseClass"] = BOMH::derivedFrom;
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
   static Function* lenFunc = Engine::instance()->getPseudoFunction("len");
   fassert( lenFunc != 0 );

   Item &value = ctx->topData();
   value.methodize(lenFunc);
}


void len_(VMContext* ctx, const Class*, void*)
{
   Item& topData = ctx->topData();
   topData = topData.len();
}


void bound(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );
}

void bound_(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );
}  

void className(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );
}


void className_(VMContext* ctx, const Class*, void* data)
{
   Item* pself;
   OpToken token( ctx, pself );

   // which means using force class inst.
   Class* cls1;
   pself->forceClassInst( cls1, data );
   token.exit(cls1->name()); // garbage this string
}

void baseClass(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );
}


void baseClass_(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );
}

//======================================================
// Clone

void clone(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );

}


void clone_(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );

}


void describe( VMContext* ctx, const Class*, void* )
{
   static Function* func = Engine::instance()->getCore()->getFunction("describe");
   fassert( func != 0 );

   Item &value = ctx->topData();
   value.methodize(func);
}


void describe_(VMContext* ctx, const Class* cls, void* data)
{
   String* target = new String;
   cls->describe( data, *target );

   Item& topData = ctx->topData();
   topData.setDeep( target->garbage() );
}


void isCallable(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );

}


void isCallable_(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );

}


void metaclass(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );

}


void metaclass_(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );

}


void ptr(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );

}

void ptr_(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );

}


void toString(VMContext* ctx, const Class*, void*)
{
   static Function* func = Engine::instance()->getCore()->getFunction("toString");
   fassert( func != 0 );

   Item &value = ctx->topData();
   value.methodize(func);
}


void toString_(VMContext* ctx, const Class* cls, void* data)
{
   cls->op_toString( ctx, data );
}


void typeId(VMContext* ctx, const Class*, void*)
{
   static Function* func = Engine::instance()->getPseudoFunction("typeId");
   fassert( func != 0 );

   Item &value = ctx->topData();
   value.methodize(func);
}


void typeId_(VMContext* ctx, const Class*, void* data)
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
   static Function* func = Engine::instance()->getPseudoFunction("compare");
   fassert( func != 0 );

   Item &value = ctx->topData();
   value.methodize(func);
}


void derivedFrom(VMContext*, const Class*, void*)
{
   fassert2( false, "Not implemented" );
}

}

}

/* end of bom.cpp */
