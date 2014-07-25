/*
   FALCON - The Falcon Programming Language.
   FILE: classmantra.cpp

   Base handler for function and classes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 18 Jun 2011 21:24:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/classes/mantra.cpp"

#include <falcon/classes/classmantra.h>
#include <falcon/error.h>
#include <falcon/mantra.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/optoken.h>
#include <falcon/itemdict.h>
#include <falcon/stdhandlers.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>
#include <falcon/module.h>
#include <falcon/stdhandlers.h>
#include <falcon/stdsteps.h>
#include <falcon/classes/classmodule.h>

#include <falcon/modspace.h>
#include <falcon/pseudofunc.h>
#include <falcon/stderrors.h>

namespace Falcon {

ClassMantra::ClassMantra():
   Class( "Mantra", FLC_ITEM_USER )
{
   m_getAttributeMethod.methodOf(this);
   m_setAttributeMethod.methodOf(this);
   m_clearPriority = 1;
}

ClassMantra::ClassMantra( const String& name, int64 type ):
   Class( name, type )
{
   m_getAttributeMethod.methodOf(this);
   m_setAttributeMethod.methodOf(this);
   m_clearPriority = 1;
}


ClassMantra::~ClassMantra()
{
}

void ClassMantra::gcMarkInstance( void* self, uint32 mark ) const
{
   static_cast<Mantra*>(self)->gcMark( mark );
}

bool ClassMantra::gcCheckInstance( void* self, uint32 mark ) const
{
   return static_cast<Mantra*>(self)->gcCheck( mark );
}

void ClassMantra::dispose( void* self ) const
{
   TRACE( "ClassMantra::dispose %p", self);

   Mantra* mantra = static_cast<Mantra*>(self);
   TRACE1( "ClassMantra::dispose -- detail: %s", mantra->locate().c_ize() );

   delete mantra;
}


void* ClassMantra::clone( void* source ) const
{
   return source;
}

void* ClassMantra::createInstance() const
{
   return 0;
}


void ClassMantra::enumerateProperties( void*, Class::PropertyEnumerator& cb ) const
{
   cb("attributes");
   cb("category" );
   cb("location");
   cb("module");
   cb("name");
}


void ClassMantra::enumeratePV( void* instance, Class::PVEnumerator& cb ) const
{
   Mantra* mantra = static_cast<Mantra*>(instance);

   Item i_name = mantra->name();
   Item i_loc = mantra->locate();
   Item i_cat = (int64) mantra->category();

   cb("category", i_cat );
   cb("location", i_loc );
   cb("name", i_name );
}


bool ClassMantra::hasProperty( void*, const String& prop ) const
{
   return
         prop == "attributes"
         || prop == "category"
         || prop == "location"
         || prop == "getAttribute"
         || prop == "module"
         || prop == "name"
         || prop == "setAttribute"
         ;
}


void ClassMantra::describe( void* instance, String& target, int, int ) const
{
   Mantra* fc = static_cast<Mantra*>(instance);
   target = "Mantra " + fc->locate();
}

//====================================================================
// Storage
//

void ClassMantra::store( VMContext*, DataWriter* stream, void* instance ) const
{
   Mantra* mantra = static_cast<Mantra*>(instance);
   TRACE1( "ClassMantra::store -- starting store mantra %s", mantra->name().c_ize());

   stream->write( mantra->name() );

   // if store
   if( mantra->module() != 0 )
   {
      stream->write( true );
      stream->write( mantra->module()->name() );
      stream->write( mantra->module()->uri() );
   }
   else {
      stream->write( false );
   }
}


void ClassMantra::restore( VMContext* ctx, DataReader* stream ) const
{
   MESSAGE1( "ClassMantra::restore -- starting restore");
   
   static Engine* eng = Engine::instance()->instance();
   
   String name;
   stream->read(name);      
   TRACE1( "ClassMantra::restore -- Restoring mantra %s", name.c_ize() );
   
   bool bHasModule;      
   stream->read(bHasModule);      
   if( bHasModule ) 
   {
      String modName, modUri;
      stream->read( modName );
      stream->read( modUri );

      TRACE2( "ClassMantra::restore -- Restoring dynamic mantra %s from %s: %s",
            name.c_ize(), modName.c_ize(), modUri.c_ize() );

      //TODO: is the main VM module space the right place?
      ModSpace* ms = ctx->process()->modSpace();
      // this might alter the context and go deep
      ms->findDynamicMantra( ctx, modUri, modName, name );
   }
   else {
      Mantra* mantra = eng->getMantra( name );
      
      // if 0, would have thrown
      fassert( mantra != 0 );
      ctx->pushData( Item( this, mantra ) );
   }      
}

//====================================================================
// Operator overloads
//


void ClassMantra::op_isTrue( VMContext* ctx, void* ) const
{
   // classes are always true
   ctx->topData().setBoolean(true);
}

void ClassMantra::op_toString( VMContext* ctx , void* item ) const
{
   Class* fc = static_cast<Class*>(item);
   String* sret = new String( "Mantra " );
   sret->append(fc->name());
   ctx->topData() = FALCON_GC_STORE( sret->handler(), sret );
}

void ClassMantra::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   Mantra* mantra = static_cast<Mantra*>(instance);

   if( prop == "attributes" )
   {
      ItemDict* dict = new ItemDict;
      uint32 size = mantra->attributes().size();
      for( uint32 i = 0; i < size; ++i ) {
         Attribute* attr = mantra->attributes().get(i);
         dict->insert( FALCON_GC_HANDLE( new String(attr->name())), attr->value() );
      }
      ctx->stackResult(1, FALCON_GC_HANDLE(dict) );
   }
   else if( prop == "category" )
   {
      ctx->stackResult(1, (int64) mantra->category() );
   }
   else if( prop == "location" )
   {
      ctx->stackResult(1, FALCON_GC_HANDLE( new String(mantra->locate())) );
   }
   else if( prop == "getAttribute" )
   {
      ctx->topData().methodize( &m_getAttributeMethod );
   }
   else if(  prop == "module" )
   {
      if( mantra->module() != 0 ) {
         static Class* clsMod = Engine::handlers()->moduleClass();
         ctx->stackResult(1, Item(clsMod, mantra->module()) );
      }
      else {
         ctx->stackResult(1, Item() );
      }
   }
   else if( prop == "name" )
   {
      ctx->stackResult(1, FALCON_GC_HANDLE( new String(mantra->name())) );
   }
   else if( prop == "setAttribute" )
   {
      ctx->topData().methodize( &m_setAttributeMethod );
   }
   else {
      Class::op_getProperty(ctx, instance, prop );
   }
}


void ClassMantra::op_summon( VMContext* ctx, void* instance, const String& message, int32 pCount, bool bOptional ) const
{
   Mantra* mantra = static_cast<Mantra*>(instance);
   Item delegated;

   if( message != "delegate" && mantra->delegates().getDelegate(message, delegated) )
   {
      ctx->opcodeParam(pCount) = delegated;
      Class* cls;
      void* inst;
      delegated.forceClassInst(cls, inst);
      cls->op_summon(ctx, inst, message, pCount, bOptional);
      return;
   }

   Class::op_summon(ctx, instance, message, pCount, bOptional);
}

void ClassMantra::delegate( void* instance, Item* target, const String& message ) const
{
   Mantra* mantra = static_cast<Mantra*>(instance);
   if( target == 0 )
   {
      mantra->delegates().clear();
   }
   else {
      mantra->delegates().setDelegate(message, *target);
   }
}

//===============================================================
// getAttribute method
//
ClassMantra::GetAttributeMethod::GetAttributeMethod():
   Function("getAttribute")
{
   signature("S");
   addParam("name");
}

ClassMantra::GetAttributeMethod::~GetAttributeMethod()
{}


void ClassMantra::GetAttributeMethod::invoke( VMContext* ctx, int32 )
{
   Item& self = ctx->self();
   fassert( self.isUser() );

   Item* i_name = ctx->param(0);
   if( ! i_name->isString() )
   {
      ctx->raiseError(paramError());
      return;
   }

   const String& attName = *i_name->asString();
   Mantra* mantra = static_cast<Mantra*>(self.asInst());
   Attribute* attr = mantra->attributes().find(attName);
   if( attr == 0 )
   {
      ctx->raiseError( new AccessError( ErrorParam(e_dict_acc, __LINE__, SRC )
            .symbol("Mantra.getAttribute")
            .module("[core]")
            .extra(attName) ) );
      return;
   }

   if( attr->value().isTreeStep() )
   {
      static const PStep* psReturn = &Engine::instance()->stdSteps()->m_returnFrameWithTop;
      ctx->pushCode( psReturn );
      ctx->pushCode( static_cast<PStep*>(attr->value().asInst()) );
   }
   else {
      ctx->returnFrame(attr->value());
   }
}

ClassMantra::SetAttributeMethod::SetAttributeMethod():
   Function("setAttribute")
{
   signature("S,[X]");
   addParam("name");
   addParam("value");
}

ClassMantra::SetAttributeMethod::~SetAttributeMethod()
{}


void ClassMantra::SetAttributeMethod::invoke( VMContext* ctx, int32 )
{
   Item& self = ctx->self();
   fassert( self.isUser() );

   Item* i_name = ctx->param(0);
   Item* i_value = ctx->param(1);
   if( ! i_name->isString() )
   {
      ctx->raiseError(paramError());
      return;
   }

   const String& attName = *i_name->asString();
   Mantra* mantra = static_cast<Mantra*>(self.asInst());

   if( i_value == 0 )
   {
      mantra->attributes().remove(attName);
   }
   else {
      Attribute* attr = mantra->attributes().find(attName);
      if( attr == 0 )
      {
         attr = mantra->attributes().add(attName);
      }

      attr->value().copyInterlocked( *i_value );
   }

   ctx->returnFrame();
}

}

/* end of metaclass.cpp */
