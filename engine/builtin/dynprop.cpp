/*
   FALCON - The Falcon Programming Language.
   FILE: dynprop.cpp

   Falcon core module -- Dynamic property handlers (set, get, has).
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 07 Mar 2013 03:52:57 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/builtin/dynprop.cpp"

#include <falcon/builtin/dynprop.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/itemid.h>
#include <falcon/itemdict.h>
#include <falcon/error.h>
#include <falcon/stderrors.h>

namespace Falcon {
namespace Ext {

Get::Get():
   PseudoFunction( "get", &m_invoke )
{
   signature("X,S");
   addParam("item");
   addParam("property");
}

Get::~Get()
{
}

void Get::invoke( VMContext* ctx, int32 )
{
   Item *elem;
   Item *i_propName;
   
   if ( ctx->isMethodic() )
   {
      elem = &ctx->self();
      i_propName = ctx->param(0);
   }
   else
   {
      elem = ctx->param(0);
      i_propName = ctx->param(1);
   }
   
   if( elem == 0 || i_propName == 0 || ! i_propName->isString() )
   {
      throw paramError();
   }

   String propName = *i_propName->asString();

   Class* cls; void* inst;
   elem->forceClassInst( cls, inst );
   ctx->returnFrame();

   ctx->pushData(Item(cls, inst));
   cls->op_getProperty( ctx, inst, propName );
}


void Get::Invoke::apply_( const PStep*, VMContext* ctx )
{
   register Item& elem = ctx->opcodeParam(1);
   register Item& i_propName = ctx->opcodeParam(0);

   if( ! i_propName.isString() )
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC).extra("X,S") );
   }

   ctx->popCode();

   Class* cls; void* inst;
   elem.forceClassInst( cls, inst );
   String propName = *i_propName.asString();
   ctx->popData();

   cls->op_getProperty(ctx, inst, propName );
}




Set::Set():
   PseudoFunction( "set", &m_invoke )
{
   signature("X,S,X");
   addParam("item");
   addParam("property");
   addParam("value");
}

Set::~Set()
{
}

void Set::invoke( VMContext* ctx, int32 )
{
   Item *elem;
   Item *i_propName;
   Item *i_value;

   if ( ctx->isMethodic() )
   {
      elem = &ctx->self();
      i_propName = ctx->param(0);
      i_value = ctx->param(1);
   }
   else
   {
      elem = ctx->param(0);
      i_propName = ctx->param(1);
      i_value = ctx->param(2);
   }

   if( elem == 0 || i_propName == 0 || ! i_propName->isString() || i_value == 0)
   {
      throw paramError();
   }

   String propName = *i_propName->asString();
   Item value = *i_value;

   Class* cls; void* inst;
   elem->forceClassInst( cls, inst );
   ctx->returnFrame();

   ctx->pushData(value);
   ctx->pushData(Item(cls, inst));
   cls->op_setProperty( ctx, inst, propName );
}


void Set::Invoke::apply_( const PStep*, VMContext* ctx )
{
   register Item& i_elem = ctx->opcodeParam(2);
   register Item& i_propName = ctx->opcodeParam(1);
   register Item& i_value = ctx->opcodeParam(0);

   if( ! i_propName.isString() )
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC).extra("X,S,X") );
   }

   ctx->popCode();

   Class* cls; void* inst;
   i_elem.forceClassInst( cls, inst );
   String propName = *i_propName.asString();

   Item elem = i_elem;
   Item value = i_value;
   ctx->popData(3);
   ctx->pushData( value );
   ctx->pushData(elem);

   cls->op_setProperty(ctx, inst, propName );
}




Has::Has():
   PseudoFunction( "has", &m_invoke )
{
   signature("X,S");
   addParam("item");
   addParam("property");
}

Has::~Has()
{
}

void Has::invoke( VMContext* ctx, int32 )
{
   Item *elem;
   Item *i_propName;

   if ( ctx->isMethodic() )
   {
      elem = &ctx->self();
      i_propName = ctx->param(0);
   }
   else
   {
      elem = ctx->param(0);
      i_propName = ctx->param(1);
   }

   if( elem == 0 || i_propName == 0 || ! i_propName->isString() )
   {
      throw paramError();
   }

   String& propName = *i_propName->asString();

   Class* cls; void* inst;
   elem->forceClassInst( cls, inst );
   bool has = cls->hasProperty(inst, propName );
   ctx->returnFrame( Item().setBoolean(has) );
}


void Has::Invoke::apply_( const PStep*, VMContext* ctx )
{
   register Item& i_elem = ctx->opcodeParam(1);
   register Item& i_propName = ctx->opcodeParam(0);

   if( ! i_propName.isString() )
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC).extra("X,S") );
   }

   ctx->popCode();

   Class* cls; void* inst;
   i_elem.forceClassInst( cls, inst );
   String propName = *i_propName.asString();
   ctx->popData();
   ctx->topData().setBoolean( cls->hasProperty(inst, propName));
}





Properties::Properties():
   PseudoFunction( "properties", &m_invoke )
{
   signature("X");
   addParam("item");
}

Properties::~Properties()
{
}

static ItemDict* internal_enumerate( Item* elem )
{
   Class* cls; void* inst;
   elem->forceClassInst( cls, inst );

   class Enumerator: public Class::PVEnumerator
   {
   public:
      Enumerator( ItemDict* dict ): m_dict(dict){}
      virtual ~Enumerator(){}

      virtual void operator()( const String& property, Item& value )
      {
         m_dict->insert( FALCON_GC_HANDLE( new String(property) ), value );
      }

   private:
      ItemDict* m_dict;
   };

   ItemDict* dict = new ItemDict;
   Enumerator rator( dict );

   cls->enumeratePV(inst, rator);
   return dict;
}

void Properties::invoke( VMContext* ctx, int32 )
{
   Item *elem;

   if ( ctx->isMethodic() )
   {
      elem = &ctx->self();
   }
   else
   {
      elem = ctx->param(0);
      if( elem == 0 )
      {
         throw paramError();
      }
   }

   ItemDict* dict = internal_enumerate(elem);
   ctx->returnFrame( FALCON_GC_HANDLE(dict) );
}


void Properties::Invoke::apply_( const PStep*, VMContext* ctx )
{
   Item* i_elem = &ctx->opcodeParam(0);
   ItemDict* dict = internal_enumerate(i_elem);
   ctx->popCode();
   ctx->topData() = FALCON_GC_HANDLE(dict);
}


}
}

/* end of dynprop.cpp */

