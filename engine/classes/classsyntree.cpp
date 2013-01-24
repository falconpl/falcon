/*
   FALCON - The Falcon Programming Language.
   FILE: classsyntree.cpp

   Base class for statement PStep handlers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 27 Dec 2011 21:39:56 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/classes/classsyntree.cpp"

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/syntree.h>
#include <falcon/vmcontext.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/symbol.h>

#include <falcon/classes/classsyntree.h>
#include <falcon/classes/classtreestep.h>
#include <falcon/classes/classsymbol.h>

#include <falcon/errors/paramerror.h>

#include "falcon/stdsteps.h"

namespace Falcon {

ClassSynTree::ClassSynTree( ClassTreeStep* parent, ClassSymbol* sym ):
   DerivedFrom( parent, "SynTree" ),
   m_classSymbol( sym )
{}
   
ClassSynTree::~ClassSynTree(){}


void* ClassSynTree::createInstance() const
{
   return new SynTree;
}


void ClassSynTree::dispose( void* instance ) const
{
   SynTree* st = static_cast<SynTree*>(instance);
   delete st;
}


void* ClassSynTree::clone( void* instance ) const
{
   SynTree* st = static_cast<SynTree*>(instance);
   return st->clone();
}


void ClassSynTree::store( VMContext* ctx, DataWriter* stream, void* instance ) const
{
   SynTree* st = static_cast<SynTree*>(instance);

   if(st->target() != 0) {
      stream->write( true );
      stream->write( st->target()->isGlobal() );
      stream->write( st->target()->name() ) ;
   }
   else {
      stream->write( false );
   }

   m_parent->store( ctx, stream, instance );
}


void ClassSynTree::restore( VMContext* ctx, DataReader* stream ) const
{
   bool hasSym;
   bool isGlobal;
   SynTree* st = new SynTree;

   stream->read( hasSym );
   if( hasSym ) {
      stream->read( isGlobal );
      String symname;
      stream->read( symname );
      Symbol* sym = Engine::getSymbol( symname, isGlobal );
      st->target(sym);
   }

   try {
      ctx->pushData( Item( this, st ) );
      // the parent wants us on top of stack.
      m_parent->restore( ctx, stream );
   }
   catch( ... ) {
      ctx->popData();
      delete st;
      throw;
   }
}

void ClassSynTree::unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   m_parent->unflatten( ctx, subItems, instance );
   SynTree* tgt = static_cast<SynTree*>(instance);
   tgt->setApply();
}


void ClassSynTree::enumerateProperties( void* instance, Class::PropertyEnumerator& cb ) const
{
   cb("target", false);
   m_parent->enumerateProperties(instance, cb);
}


void ClassSynTree::enumeratePV( void* instance, Class::PVEnumerator& cb ) const
{
   SynTree* st = static_cast<SynTree*>(instance);
   
   Item i_selector, i_target;
   if( st->target() != 0 ) i_target.setUser( m_classSymbol, st->target() );
   
   cb("target", i_target ); 
   m_parent->enumeratePV(instance, cb);
}


bool ClassSynTree::hasProperty( void* instance, const String& prop ) const
{
   return  
         prop == "target" 
         || m_parent->hasProperty( instance, prop );
}

void ClassSynTree::op_getProperty(VMContext* ctx, void* instance, const String& prop) const
{
   SynTree* st = static_cast<SynTree*>(instance);
   
   if( prop == "target" )
   {
      Item i_target;
      if( st->target() != 0 ) i_target.setUser( m_classSymbol, st->target() );
      ctx->stackResult(1, i_target );
   }
   else {
      m_parent->op_getProperty( ctx, instance, prop );
   }
}


void ClassSynTree::op_setProperty( VMContext* ctx, void* instance, const String& prop) const
{
   SynTree* st = static_cast<SynTree*>(instance);
   Item& source = ctx->opcodeParam(0);
   if ( prop == "target" ) {
      // is a symbol?
      Class* cls; void* src;
      if( source.asClassInst( cls, src ) && cls->isDerivedFrom( m_classSymbol ) )
      {
         // ok, we can go.
         st->target( static_cast<Symbol*>(src) );
         ctx->stackResult(3, source);
         return;
      }
      
      // If we're here, it's not an expression
      throw new ParamError( ErrorParam(e_param_type, __LINE__, SRC )
            .origin( ErrorParam::e_orig_vm)
            .extra( "Symbol" ) );
   }
   else {
      m_parent->op_setProperty( ctx, instance, prop );
   }
}


void ClassSynTree::op_call(VMContext* ctx, int pcount, void* instance) const
{
   SynTree* tree = static_cast<SynTree*>(instance);
   ctx->pushCode( tree );
   ctx->stackResult(pcount+1, Item());
}

}

/* end of classsyntree.cpp */
