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

void len(VMachine* vm, const Class*, void*)
{
   static Function* lenFunc = Engine::instance()->getPseudoFunction("len");
   fassert( lenFunc != 0 );

   Item &value = vm->currentContext()->topData();
   value.methodize(lenFunc);
}


void len_(VMachine* vm, const Class*, void*)
{
   Item& topData = vm->currentContext()->topData();
   topData = topData.len();
}


void bound(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );
}

void bound_(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );
}  

void className(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );
}


void className_(VMachine* vm, const Class* cls, void* data)
{
   Item* pself;
   OpToken token( vm, pself );

   // which means using force class inst.
   Class* cls1;
   pself->forceClassInst( cls1, data );
   token.exit(cls1->name()); // garbage this string
}

void baseClass(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );
}


void baseClass_(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );
}

//======================================================
// Clone

void clone(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );

}


void clone_(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );

}


void describe(VMachine* vm, const Class* cls, void* data)
{
   static Function* func = Engine::instance()->getCore()->getFunction( "describe");
   fassert( func != 0 );

   Item &value = vm->currentContext()->topData();
   value.methodize(func);
}


void describe_(VMachine* vm, const Class* cls, void* data)
{
   String* target = new String;
   cls->describe( data, *target );

   Item& topData = vm->currentContext()->topData();
   topData.setDeep( target->garbage() );
}


void isCallable(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );

}


void isCallable_(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );

}


void metaclass(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );

}


void metaclass_(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );

}


void ptr(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );

}

void ptr_(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );

}


void toString(VMachine* vm, const Class* cls, void* data)
{
   static Function* func = Engine::instance()->getCore()->getFunction("toString");
   fassert( func != 0 );

   Item &value = vm->currentContext()->topData();
   value.methodize(func);
}


void toString_(VMachine* vm, const Class* cls, void* data)
{
   cls->op_toString( vm, data );
}


void typeId(VMachine* vm, const Class* cls, void* data)
{
   static Function* func = Engine::instance()->getPseudoFunction("typeId");
   fassert( func != 0 );

   Item &value = vm->currentContext()->topData();
   value.methodize(func);
}


void typeId_(VMachine* vm, const Class* cls, void* data)
{
   Item* pself;
   OpToken token( vm, pself );

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



void compare(VMachine* vm, const Class* cls, void* data)
{
   static Function* func = Engine::instance()->getPseudoFunction("compare");
   fassert( func != 0 );

   Item &value = vm->currentContext()->topData();
   value.methodize(func);
}


void derivedFrom(VMachine* vm, const Class* cls, void* data)
{
   fassert2( false, "Not implemented" );
}

}

}

/* end of bom.cpp */
