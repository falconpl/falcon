/*
   FALCON - The Falcon Programming Language.
   FILE: exprinherit.cpp

   Structure holding information about inheritance in a class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jun 2011 13:35:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/psteps/exprinherit.h>
#include <falcon/expression.h>
#include <falcon/vmcontext.h>
#include <falcon/class.h>
#include <falcon/synclasses.h>


#include <falcon/string.h>
#include "exprvector_private.h"


namespace Falcon
{

ExprInherit::ExprInherit( int line, int chr ):
   ExprVector( line, chr ),
   m_base(0),
{
   FALCON_DECLARE_SYN_CLASS( expr_inherit )
   apply = apply_;
}

ExprInherit::ExprInherit( const String& name, int line, int chr ):
   ExprVector( line, chr )
   m_base(0),
   m_name( name )
{
   FALCON_DECLARE_SYN_CLASS( expr_inherit )
   apply = apply_;
}

ExprInherit::ExprInherit( Class* base, int line, int chr ):
   ExprVector( line, chr ),
   m_base( base ),
   m_name( base->name() )
{
   FALCON_DECLARE_SYN_CLASS( expr_inherit )
   apply = apply_;
}
   
ExprInherit::ExprInherit( const ExprInherit& other ):
   ExprVector( other ),
   m_base( other.m_base ),
   m_name( other.m_name )
{
   apply = apply_;
}
  
ExprInherit::~ExprInherit()
{
}

void ExprInherit::describeTo( String& target, int depth ) const
{   
   String prefix = String(" ").replicate( depth * depthIndent );
   
   if( _p->m_exprs.empty() )
   {
      target = prefix + m_name;
   }
   else
   {
      target = prefix + m_name + "(";     
      String temp;
      ExprVector_Private::ExprVector::const_iterator iter = _p->m_exprs.begin();
      while(  _p->m_exprs.end() != iter )
      {
         Expression* param = *iter;
         if( temp.size() >  0 )
         {
            temp += ", ";
         }
         // keep same depth
         temp += param->describe( depth );
         ++iter;
      }

      target += temp + ")";
   }
}

void ExprInherit::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprInherit* self = static_cast<const ExprInherit*>(ps);
   fassert( self->m_base != 0 );
   
   // we need to "produce" the parameters, were any of it.
   CodeFrame& cf = ctx->currentCode();
   int& seqId = cf.m_seqId;
   const ExprVector_Private::ExprVector& exprs = self->_p->m_exprs;   
   int size = (int) exprs.size();   
   
   TRACE1("Apply with %d/%d parameters", seqId, size );
   
   while( seqId < size )
   {
      Expression* exp = exprs[seqId++];
      if( ctx->stepInYield( exp, cf ) )
      {
         return;
      }
   }
   
   // we have expanded all the parameters. Go for init the class.
   ctx->popCode();
   Item* iinst = ctx->opcodeParam(size+1);
   // The creation process must have given the instance to us right before
   // -- the parameters were created.
   fassert( iinst->isClass() );
   fassert( iinst->asClass()->derivedFrom(self->m_base) );
   void* instance = iinst->asClass()->getParentData( self->m_base, iinst->asInst());
   
   // invoke the init operator directly
   if( self->m_base->op_init( ctx, instance, size ) )
   {
      // It's deep.
      return;
   }
   // we're in charge of popping the parameters.
   ctx->popData(size);
}

/*

void Inheritance::IRequirement::onResolved(   
         const Module* source, const Symbol* srcSym, Module* tgt, Symbol* )
{
   const Item* value;
   
   if( (value = srcSym->getValue( 0 )) == 0 || ! value->isClass() )
   {
      // the symbol is not a class?   
      throw new CodeError( ErrorParam( e_inv_inherit ) 
         .module( source == 0 ? "<internal>" : source->uri() )
         .symbol( srcSym->name() )
         .line( m_owner->sourceRef().line())
         .chr( m_owner->sourceRef().chr())
         .origin(ErrorParam::e_orig_linker));
   }

   // Ok, we have a valid class.
   Class* newParent = static_cast<Class*>(value->asInst());
   m_owner->parent( newParent );
   Class* cls = m_owner->m_owner;
   // is the owner class a Falcon class?
   if( cls->isFalconClass() )
   {
      // then, see if we can link it.
      FalconClass* falcls = static_cast<FalconClass*>(cls);
      if( falcls->missingParents() == 0 && tgt != 0 )
      {
         // ok, the parent that has been found now was the last one.
         tgt->completeClass( falcls );
      }
   }
}
  */
 
}

/* end of exprinherit.cpp */
