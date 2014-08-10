/*
   FALCON - The Falcon Programming Language.
   FILE: metafalconclass.cpp

   Handler for classes defined by a Falcon script.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 10 Mar 2012 23:27:16 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#undef SRC
#define SRC "engine/metafalconclass.cpp"

#include <falcon/classes/metafalconclass.h>
#include <falcon/hyperclass.h>
#include <falcon/falconclass.h>
#include <falcon/engine.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/storer.h>
#include <falcon/stderrors.h>
#include <falcon/itemdict.h>
#include <falcon/stdhandlers.h>

namespace Falcon
{

MetaFalconClass::MetaFalconClass()
{
   m_name = "FalconClass";
}

MetaFalconClass::~MetaFalconClass()
{
}

const Class* MetaFalconClass::handler() const
{
   static Class* cls = Engine::handlers()->metaFalconClass();   
   return cls;
}

void MetaFalconClass::store( VMContext*, DataWriter* wr, void* inst ) const
{
   FalconClass* fcls = static_cast<FalconClass*>(inst);
   
   wr->write( fcls->name() );
   // if we're storing our own module, then we should save our unconstructed status.
   /*Storer* storer = ctx->getTopStorer();
   bool isConstructed = storer == 0 || storer->topData() != fcls->module();*/
   fcls->storeSelf( wr );
}


void MetaFalconClass::restore( VMContext* ctx, DataReader* rd ) const
{
   String name;
   rd->read( name );
   
   FalconClass* fcls = new FalconClass( name );
   try {
      fcls->restoreSelf( rd );
      ctx->pushData( Item( this, fcls ) );
   }
   catch(...) {
      delete fcls;
      throw;
   }
}


void MetaFalconClass::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   FalconClass* fcls = static_cast<FalconClass*>(instance);
   /*
   Storer* storer = ctx->getTopStorer();
   bool isConstructed = storer == 0 || storer->topData() != fcls->module();
   */
   fcls->flattenSelf( subItems );
}
   

void MetaFalconClass::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   FalconClass* fcls = static_cast<FalconClass*>(instance);
   fcls->unflattenSelf( subItems );
}


bool MetaFalconClass::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   static Class* classParentship = static_cast<Class*>(
            Engine::instance()->getMantra("Syn.Parentship", Mantra::e_c_class));
   fassert( classParentship != 0 );
   
   Item* operands = ctx->opcodeParams( pcount );
   // Class constructor:
   // Class( name, members, parentship )
   
   if( pcount < 1 || ! operands->isString() )
   {
      throw new ParamError( ErrorParam(e_inv_params, __LINE__,SRC)
            .origin(ErrorParam::e_orig_runtime)
            .extra("S,[D],[Parentship]") );
   }
   
   
   String* name = operands->asString();
   ItemDict* members = 0;
   ExprParentship* ep = 0;
   
   if( pcount > 1 )
   {
      register Item* iDict = operands + 1;
      if( ! iDict->isDict() ) {
         throw new ParamError( ErrorParam(e_inv_params, __LINE__,SRC)
            .origin(ErrorParam::e_orig_runtime)
            .extra("S,[D],[Parentship]") );
      }
      members = iDict->asDict();
   }
   
   if( pcount > 2 )
   {
      register Item* iParent = operands + 2;
      Class* cls;
      void* data;
      
      if( ! iParent->asClassInst(cls, data) || ! cls->isDerivedFrom(classParentship) ) {
         throw new ParamError( ErrorParam(e_inv_params, __LINE__,SRC)
            .origin(ErrorParam::e_orig_runtime)
            .extra("S,[D],[Parentship]") );
      }
      ep = static_cast<ExprParentship*>( cls->getParentData(classParentship, data ) );
   }
   
   // we have all the needed parameters.
   FalconClass* fcls = static_cast<FalconClass*>( instance );
   fcls->name( *name );
   if( ep != 0 )
   {
      fcls->setParentship( ep );
   }
   
   if( members != 0 )
   {
      // process the members.
      class Rator: public ItemDict::Enumerator {
      public:
         Rator( FalconClass* theClass ):
            m_flc( theClass )
         {}
            
         virtual void operator()( const Item& key, Item& value )
         {
            if( ! key.isString() )
            {
               throw new ParamError( ErrorParam(e_param_type, __LINE__,SRC)
                  .origin(ErrorParam::e_orig_runtime)
                  .extra("Member dictionary must have strings for keys") ); 
            }
            
            String* propName = key.asString();
            if( *propName == "init" )
            {
               Function* initFunc;
               if( ! value.isFunction() || 
                  (initFunc = value.asFunction())->category() != Mantra::e_c_synfunction )
               {
                  throw new ParamError( ErrorParam(e_param_type, __LINE__,SRC)
                     .origin(ErrorParam::e_orig_runtime)
                     .extra("Init member must be a syntactic function") ); 
               }
               
               SynFunc* sf = static_cast<SynFunc*>(initFunc);
               sf->methodOf( m_flc );
               sf->setConstructor();
               m_flc->setConstructor(sf);
            }
            else {
               if( value.isFunction() )
               {
                  // force the name
                  Function* f = value.asFunction();
                  f->methodOf(m_flc);
                  m_flc->addMethod( *propName, f );
               }
               else {
                  // TODO: state values
                  m_flc->addProperty( *propName, value );
               }
            }            
         }
         
      private:
         FalconClass* m_flc;
      };
      Rator rator( fcls );
      
      members->enumerate( rator );
   }
   
   if( ! fcls->construct(ctx) )
   {
      // we need to hyperconstruct it -- and this means changing the instance.
      HyperClass* hcls = fcls->hyperConstruct();
      ctx->popData(pcount);
      ctx->topData().setUser( hcls->handler(), hcls );
      return true;
   }
   return false;
}

   
void* MetaFalconClass::createInstance() const
{
   return new FalconClass("#anonymous");
}

}

/* end of metafalconclass.cpp */
