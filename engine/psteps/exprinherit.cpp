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
#include <falcon/error.h>
#include <falcon/errors/codeerror.h>
#include <falcon/itemarray.h>
#include <falcon/classes/classrequirement.h>

#include <falcon/symbol.h>
#include <falcon/module.h>
#include <falcon/falconclass.h>

#include <falcon/string.h>
#include "exprvector_private.h"


namespace Falcon
{

ExprInherit::ExprInherit( int line, int chr ):
   ExprVector( line, chr ),
   m_base(0),
   m_bHadRequirement(false)
{
   FALCON_DECLARE_SYN_CLASS( expr_inherit )
   apply = apply_;
}

ExprInherit::ExprInherit( const String& name, int line, int chr ):
   ExprVector( line, chr ),
   m_base(0),
   m_name( name ),
   m_bHadRequirement(false)
{
   FALCON_DECLARE_SYN_CLASS( expr_inherit )
   apply = apply_;
}

ExprInherit::ExprInherit( Class* base, int line, int chr ):
   ExprVector( line, chr ),
   m_base( base ),
   m_name( base->name() ),
   m_bHadRequirement(false)
{
   FALCON_DECLARE_SYN_CLASS( expr_inherit )
   apply = apply_;
}
   
ExprInherit::ExprInherit( const ExprInherit& other ):
   ExprVector( other ),
   m_base( other.m_base ),
   m_name( other.m_name ),
   m_bHadRequirement(false)
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
   Item* iinst = ctx->opcodeParams(size+1);
   // The creation process must have given the instance to us right before
   // -- the parameters were created.
   fassert( iinst->isClass() );
   fassert( iinst->asClass()->isDerivedFrom(self->m_base) );
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


Requirement* ExprInherit::makeRequirement( Class* target )
{
   m_bHadRequirement = true;
   return new IRequirement( this, target );
}


void ExprInherit::IRequirement::onResolved( const Module* source, const Symbol* srcSym,  
      Module* tgt, Symbol* )
{
   const Item* value;
   
   if( (value = srcSym->getValue( 0 )) == 0 || ! value->isClass() )
   {
      // the symbol is not a class?   
      throw new CodeError( ErrorParam( e_inv_inherit ) 
         .module( source == 0 ? "<internal>" : source->uri() )
         .symbol( srcSym->name() )
         .line( m_owner->sr().line())
         .chr( m_owner->sr().chr())
         .origin(ErrorParam::e_orig_linker));
   }

   // Ok, we have a valid class.
   Class* newParent = static_cast<Class*>(value->asInst());
   m_owner->base( newParent );
   
   // is the owner class a Falcon class?
   if( m_target->isFalconClass() )
   {
      // then, see if we can link it.
      FalconClass* falcls = static_cast<FalconClass*>(m_target);
      if( falcls->missingParents() == 0 && tgt != 0 )
      {
         // ok, the parent that has been found now was the last one.
         tgt->completeClass( falcls );
      }
   }
}



class ExprInherit::IRequirement::ClassIRequirement: public ClassRequirement
{
public:
   ClassIRequirement():
      ClassRequirement("$IRequirement")
   {}

   virtual ~ClassIRequirement() {}
   
   virtual void store( VMContext*, DataWriter* stream, void* instance ) const
   {
      IRequirement* s = static_cast<IRequirement*>(instance);
      s->store( stream );
   }
   
   virtual void flatten( VMContext*, ItemArray& subItems, void* instance ) const
   {
      static Class* metaClass = Engine::instance()->metaClass();
      IRequirement* s = static_cast<IRequirement*>(instance);
      
      subItems.resize(2);
      subItems[0] = Item( s->m_owner->cls(), s->m_owner );
      subItems[1] = Item( metaClass, s->m_target );
   }
   
   virtual void unflatten( VMContext*, ItemArray& subItems, void* instance ) const
   {
      IRequirement* s = static_cast<IRequirement*>(instance);
      fassert( subItems.length() == 2 );
   
      s->m_owner = static_cast<ExprInherit*>(subItems[0].asInst());
      s->m_target = static_cast<Class*>( subItems[1].asInst() );
   }
   
   virtual void restore( VMContext*, DataReader* stream, void*& empty ) const
   {
      IRequirement* s = 0;
      try {
         s = new IRequirement(0,0);
         s->restore(stream);
         empty = s;
      }
      catch( ... )
      {
         delete s;
         throw;
      }
   }
   
   void describe( void* instance, String& target, int, int ) const
   {
      IRequirement* s = static_cast<IRequirement*>(instance);
      if( s->m_owner == 0 )
      {
         target = "<Blank IRequirement>";
      }
      else {
         target = "IRequirement for \"" + s->name() + "\"";
      }
   }
};


Class* ExprInherit::IRequirement::cls() const
{
   static Class* theClass = new ClassIRequirement;
   static bool bRegistered = false;

   // a double registration in MT context has no adverse effect.
   // we just need to have it in the engine to let it to delete it at end.
   if( ! bRegistered ) {
      bRegistered = true;
      Engine::instance()->registerClass(theClass);
   }

   return theClass;
}

}

/* end of exprinherit.cpp */
