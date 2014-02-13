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
#include <falcon/expression.h>
#include <falcon/vmcontext.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/symbol.h>

#include <falcon/classes/classsyntree.h>
#include <falcon/classes/classtreestep.h>
#include <falcon/classes/classsymbol.h>

#include <falcon/engine.h>
#include <falcon/stdhandlers.h>
#include <falcon/stderrors.h>

#include "falcon/stdsteps.h"

namespace Falcon {

ClassSynTree::ClassSynTree( ClassTreeStep* parent, ClassSymbol* sym ):
   ClassTreeStep( "SynTree" ),
   m_classSymbol( sym )
{
   setParent( parent );
}

ClassSynTree::ClassSynTree( const String& name, ClassSymbol* sym ):
   ClassTreeStep( name ),
   m_classSymbol( sym )
{
   if( m_classSymbol == 0)
   {
      m_classSymbol = static_cast<const ClassSymbol*>(Engine::instance()->stdHandlers()->symbolClass());
   }
}

   
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
      stream->write( st->target()->name() ) ;
   }
   else {
      stream->write( false );
   }

   ClassTreeStep::store( ctx, stream, instance );
}


void ClassSynTree::restore( VMContext* ctx, DataReader* stream ) const
{
   bool hasSym;

   SynTree* st = new SynTree;
   ctx->pushData( Item( this, st ) );

   try {
      stream->read( hasSym );
      if( hasSym ) {
         String symname;
         stream->read( symname );
         const Symbol* sym = Engine::getSymbol( symname );
         st->target(sym);
      }

      // the parent wants us on top of stack.
      ClassTreeStep::restore( ctx, stream );
   }
   catch( ... ) {
      ctx->popData();
      delete st;
      throw;
   }
}

void ClassSynTree::unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   ClassTreeStep::unflatten( ctx, subItems, instance );
}


void ClassSynTree::enumerateProperties( void* instance, Class::PropertyEnumerator& cb ) const
{
   cb("target" );
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


bool ClassSynTree::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   SynTree* tree = static_cast<SynTree*>(instance);
   // Selector, [target], TreeStep...
   tree->setInGC();

   if( pcount == 0 )
   {
      // let the MetaClass remove our parameters
      return false;
   }

   Item* params = ctx->opcodeParams(pcount);
   // we convert it now, but won't use the item till certain of what it is.
   TreeStep* first = static_cast<TreeStep*>(params->asInst());
   // the first item is the selector, and can be nil.
   if( params->isNil() )
   {
      // it's ok, we have nothing to do.
   }
   else if( params->type() != FLC_CLASS_ID_TREESTEP || first->category() != TreeStep::e_cat_expression )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_inv_params, .extra("Expression,..."));
   }
   else
   {
      if ( ! tree->selector(first) )
      {
         throw FALCON_SIGN_XERROR(ParamError, e_param_range, .extra("Parented expression at 0"));
      }
   }

   for( int i = 1; i < pcount; ++i )
   {
      Item* param = params + i;

      // idem, we convert before checking the type, but use only when sure
      TreeStep* child = static_cast<TreeStep*>(param->asInst());
      if( param->type() == FLC_CLASS_ID_SYMBOL )
      {
         if( tree->target() != 0 )
         {
            throw FALCON_SIGN_XERROR(ParamError, e_param_range, .extra(String("Target symbol already given at ").N(i)));
         }
         tree->target(static_cast<Symbol*>(param->asInst()));
      }
      else if( param->type() != FLC_CLASS_ID_TREESTEP )
      {
         throw FALCON_SIGN_XERROR(ParamError, e_inv_params, .extra(String("Not a TreeStep at ").N(i)));
      }

      else if( ! tree->append(child) )
      {
         throw FALCON_SIGN_XERROR(ParamError, e_param_range, .extra(String("Parented expression at ").N(i)));
      }
   }

   // let the MetaClass remove our parameters
   return false;
}

void ClassSynTree::op_call(VMContext* ctx, int pcount, void* instance) const
{
   static StdSteps* steps = Engine::instance()->stdSteps();
   SynTree* tree = static_cast<SynTree*>(instance);

   // We don't need parameters.
   ctx->popData(pcount+1);
   // this is a local frame.
   ctx->pushCode(&steps->m_localFrame);

   ctx->pushCode( tree );
}

}

/* end of classsyntree.cpp */
