/*
   FALCON - The Falcon Programming Language.
   FILE: classsynfunc.h

   Syntree based function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 26 Feb 2012 01:10:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classsynfunc.cpp"

#include <falcon/classes/classsynfunc.h>
#include <falcon/itemid.h>
#include <falcon/fassert.h>
#include <falcon/synfunc.h>
#include <falcon/stdhandlers.h>

#include <falcon/engine.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/itemarray.h>
#include <falcon/statement.h>

namespace Falcon
{


ClassSynFunc::ClassSynFunc():
ClassFunction( "SynFunc", FLC_CLASS_ID_FUNC )
{}

ClassSynFunc::~ClassSynFunc()
{}
   

Class* ClassSynFunc::getParent( const String& name ) const
{
   static Class* cls = Engine::handlers()->functionClass();
   
   if( name == cls->name() ) return cls;
   return 0;
}

bool ClassSynFunc::isDerivedFrom( const Class* parent ) const
{
   static Class* cls = Engine::handlers()->functionClass();
   
   return parent == this || cls->isDerivedFrom(parent);
}

void ClassSynFunc::enumerateParents( ClassEnumerator& cb ) const
{
   static Class* cls = Engine::handlers()->functionClass();
   
   cb( cls );
}

void ClassSynFunc::enumerateProperties( void* instance, Class::PropertyEnumerator& cb ) const
{
   static Class* cls = Engine::handlers()->functionClass();
   cb("code");
   cls->enumerateProperties(instance, cb);
}

void ClassSynFunc::enumeratePV( void* instance, Class::PVEnumerator& cb ) const
{
   static Class* cls = Engine::handlers()->functionClass();
   cls->enumeratePV(instance, cb);
}

bool ClassSynFunc::hasProperty( void* instance, const String& prop ) const
{
   static Class* cls = Engine::handlers()->functionClass();

   return
            prop == "code"
            ||  cls->hasProperty(instance, prop);
}

void* ClassSynFunc::getParentData( const Class* parent, void* data ) const
{
   static Class* cls = Engine::handlers()->functionClass();
   
   if( parent == cls || parent == this ) return data;
   return 0;
}


void ClassSynFunc::store( VMContext*, DataWriter* stream, void* instance ) const
{
   SynFunc* synfunc = static_cast<SynFunc*>(instance);

   TRACE1("ClassSynFunc::store %s", synfunc->name().c_ize() );
   // saving the overall data.
   stream->write( synfunc->name() );
   stream->write( synfunc->declaredAt() );
   stream->write( synfunc->isConstructor() );
   stream->write( synfunc->isEta() );
   stream->write( synfunc->signature() );
   
   // now we got to save the function parameter table.
   synfunc->parameters().store( stream );
   synfunc->locals().store( stream );
   synfunc->closed().store( stream );

   // and the attributes table
   synfunc->attributes().store(stream);
}

void ClassSynFunc::restore(VMContext* ctx, DataReader* stream) const
{
   bool bConstructor, bEta;
   int line;
   String name, signature;
   
   stream->read( name );
   TRACE1("ClassSynFunc::restore %s", name.c_ize() );

   stream->read( line );
   stream->read( bConstructor );
   stream->read( bEta );
   stream->read( signature );

   SynFunc* synfunc = new SynFunc( name, 0, line );
   synfunc->setEta( bEta );
   synfunc->signature( signature );
   if( bConstructor )
   {
      synfunc->setConstructor();
   }

   try {
      synfunc->parameters().restore( stream );
      synfunc->locals().restore( stream );
      synfunc->closed().restore( stream );

      // and the attributes table
      synfunc->attributes().restore(stream);

      ctx->pushData( Item(this, synfunc) );
   }
   catch( ... ) {
      delete synfunc;
      throw;
   }
}


void ClassSynFunc::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   SynFunc* synfunc = static_cast<SynFunc*>(instance);
      
   subItems.reserve(
            synfunc->syntree().size()
            + synfunc->attributes().size() * 2
            + 1);

   TRACE1("ClassSynFunc::flatten %s - %ld syntree elements", synfunc->name().c_ize(), (long)synfunc->syntree().size() );
   for( uint32 i = 0; i < synfunc->syntree().size(); ++i ) {
      TreeStep* stmt = synfunc->syntree().at(i);
      Class* synClass = stmt->handler();
      subItems.append(Item( synClass, stmt ) );
   }

   subItems.append(Item());
   synfunc->attributes().flatten( subItems );
}


void ClassSynFunc::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{    
   SynFunc* synfunc = static_cast<SynFunc*>(instance);
   // first restore the symbol table.
   TRACE1("ClassSynFunc::unflatten %s - %d syntree elements", synfunc->name().c_ize(), subItems.length() );

   uint32 i = 0;
   for( ; i < subItems.length(); ++i ) {
      Class* cls = 0;
      void* data = 0;
      if( ! subItems[i].asClassInst(cls,data) )
      {
         // we found the nil separator
         break;
      }

#ifndef NDEBUG
      static Class* tsc = Engine::handlers()->treeStepClass();
      fassert2( cls != 0, "Serialized instances are not classes" );
      fassert2( cls->isDerivedFrom( tsc ), "Serialized instances are not treesteps" );
#endif

      synfunc->syntree().append( static_cast<TreeStep*>(data) );
   }

   // this if may be an overkill
   ++i; // remove the extra nil
   if( i < subItems.length() )
   {
      synfunc->attributes().unflatten(subItems, i);
   }
}


void ClassSynFunc::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   static Class* cls = Engine::handlers()->functionClass();
   SynFunc* func = static_cast<SynFunc*>(instance);

   if( prop == "code" )
   {
      ctx->stackResult(1, Item(func->syntree().handler(), &func->syntree()) );
   }
   else {
      cls->op_getProperty( ctx, instance, prop );
   }
}

}

/* end of classsynfunc.cpp */





