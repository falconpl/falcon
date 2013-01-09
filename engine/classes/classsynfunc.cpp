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
   Class* cls = Engine::instance()->functionClass();
   
   if( name == cls->name() ) return cls;
   return 0;
}

bool ClassSynFunc::isDerivedFrom( const Class* parent ) const
{
   Class* cls = Engine::instance()->functionClass();
   
   return parent == cls || parent == this;
}

void ClassSynFunc::enumerateParents( ClassEnumerator& cb ) const
{
   Class* cls = Engine::instance()->functionClass();
   
   cb( cls, true );
}


void* ClassSynFunc::getParentData( Class* parent, void* data ) const
{
   Class* cls = Engine::instance()->functionClass();
   
   if( parent == cls || parent == this ) return data;
   return 0;
}



void ClassSynFunc::store( VMContext*, DataWriter* stream, void* instance ) const
{
   SynFunc* synfunc = static_cast<SynFunc*>(instance);
   // saving the overall data.
   stream->write( synfunc->name() );
   stream->write( synfunc->declaredAt() );
   stream->write( synfunc->isPredicate() );
   stream->write( synfunc->isEta() );
   stream->write( synfunc->signature() );
   
   // now we got to save the function parameter table.
   synfunc->variables().store( stream );
}

void ClassSynFunc::restore(VMContext* ctx, DataReader* stream) const
{
   bool bPred, bEta;
   int line;
   String name, signature;
   int32 pcount;
   
   stream->read( name );
   stream->read( line );
   stream->read( bPred );
   stream->read( bEta );
   stream->read( signature );
   stream->read(pcount);

   SynFunc* synfunc = new SynFunc( name, 0, line );
   ctx->pushData( FALCON_GC_STORE(this, synfunc) );
   
   synfunc->setPredicate( bPred );
   synfunc->setEta( bEta );
   synfunc->signature( signature );

   synfunc->variables().restore( stream );
}


void ClassSynFunc::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   SynFunc* synfunc = static_cast<SynFunc*>(instance);
      
   for( uint32 i = 0; i < synfunc->syntree().size(); ++i ) {
      Statement* stmt = synfunc->syntree().at(i);
      Class* synClass = stmt->handler();
      subItems.append(Item( synClass, stmt ) );
   }
}


void ClassSynFunc::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{    
   SynFunc* synfunc = static_cast<SynFunc*>(instance);
   // first restore the symbol table.
   
   for( uint32 i = 0; i < subItems.length(); ++i ) {
      Class* cls = 0;
      void* data = 0;
      subItems[i].asClassInst(cls,data);
      
#ifndef NDEBUG
      static Class* stmtClass = Engine::instance()->statementClass();
      fassert2( cls != 0, "Serialized instances are not classes" );
      fassert2( cls->isDerivedFrom( stmtClass ), "Serialized instances are not statements" );               
#endif
      
      synfunc->syntree().append( static_cast<Statement*>(data) );
   }
}

}

/* end of classsynfunc.cpp */





