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

GetP::GetP():
   PseudoFunction( "getp", &m_invoke )
{
   signature("X,S");
   addParam("item");
   addParam("property");
}

GetP::~GetP()
{
}

void GetP::invoke( VMContext* ctx, int32 )
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


void GetP::Invoke::apply_( const PStep*, VMContext* ctx )
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




SetP::SetP():
   PseudoFunction( "setp", &m_invoke )
{
   signature("X,S,X");
   addParam("item");
   addParam("property");
   addParam("value");
}

SetP::~SetP()
{
}

void SetP::invoke( VMContext* ctx, int32 )
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


void SetP::Invoke::apply_( const PStep*, VMContext* ctx )
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




HasP::HasP():
   PseudoFunction( "hasp", &m_invoke )
{
   signature("X,S");
   addParam("item");
   addParam("property");
}

HasP::~HasP()
{
}

void HasP::invoke( VMContext* ctx, int32 )
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


void HasP::Invoke::apply_( const PStep*, VMContext* ctx )
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





Approp::Approp():
   Function( "approp" )
{
   signature("X,D");
   addParam("item");
   addParam("props");
}

Approp::~Approp()
{
}

void Approp::invoke( VMContext* ctx, int32 )
{
   class PStepNext: public PStep
   {
   public:
      PStepNext(){ apply = apply_; }
      ~PStepNext() {}
      virtual void describeTo( String& target ) const { target = "Approp::PStepNext"; }

      static void apply_( const PStep*, VMContext* ctx )
      {
         int64 count = ctx->local(0)->asInteger();
         if( count == 0 )
         {
            ctx->returnFrame(*ctx->local(1));
            return;
         }

         // prepare for next call
         ctx->local(0)->setInteger(count-1);

         // remove the result of the previous operation
         ctx->popData();

         // get the required property
         String propName = *ctx->topData().asString();
         ctx->popData();

         // invoke set property of our item.
         Item* elem = ctx->local(1);
         Class* cls = 0;
         void* udata = 0;
         elem->forceClassInst( cls, udata );
         cls->op_setProperty( ctx, udata, propName );
      }

   };

   static PStepNext stepAfterInvoke;

   Item *elem, *i_dict;

   if ( ctx->isMethodic() )
   {
      elem = &ctx->self();
      i_dict = ctx->param(0);
   }
   else
   {
      elem = ctx->param(0);
      i_dict = ctx->param(1);
      if( elem == 0 )
      {
         throw paramError(__LINE__, SRC);
      }
   }

   if( i_dict == 0 || ! i_dict->isDict() )
   {
      throw paramError(__LINE__, SRC);
   }

   ItemDict* dict = i_dict->asDict();
   ctx->addLocals(2);
   *ctx->local(1) = *elem;

   // guard against read changes.
   {
      ConcurrencyGuard::Reader rg( ctx, dict->guard() );
      *ctx->local(0) = (int64) dict->size();


      // prepare the stack
      class Rator: public ItemDict::Enumerator
      {
      public:
         Rator( VMContext* ctx ): m_ctx(ctx) {}
         virtual ~Rator() {}
         virtual void operator()( const Item& key, Item& value )
         {
            if( ! key.isString() )
            {
               throw FALCON_SIGN_XERROR( ParamError, e_param_type, .extra("All the keys must be strings") );
            }

            Item copy = *m_ctx->local(1);
            m_ctx->pushData(value);
            m_ctx->pushData(copy);
            m_ctx->pushData(key);
         }

      private:
         VMContext* m_ctx;
      }
      rator(ctx);

      dict->enumerate(rator);
   }

   // push a nil item as the first value to be discarded
   ctx->pushData(Item());
   ctx->stepIn(&stepAfterInvoke);
}


}
}

/* end of dynprop.cpp */

