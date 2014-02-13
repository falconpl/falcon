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
#include <falcon/stderrors.h>
#include <falcon/itemarray.h>
#include <falcon/stdhandlers.h>
#include <falcon/textwriter.h>

#include <falcon/symbol.h>
#include <falcon/module.h>
#include <falcon/falconclass.h>

#include <falcon/datareader.h>
#include <falcon/datawriter.h>

#include <falcon/string.h>
#include "exprvector_private.h"


namespace Falcon
{

ExprInherit::ExprInherit( int line, int chr ):
   ExprVector( line, chr ),
   m_base(0),
   m_symbol(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_inherit )
   apply = apply_;
   m_trait = Expression::e_trait_inheritance;
}

ExprInherit::ExprInherit( const String& name, int line, int chr ):
   ExprVector( line, chr ),
   m_base(0),
   m_symbol( Engine::getSymbol(name) )
{
   FALCON_DECLARE_SYN_CLASS( expr_inherit )
   apply = apply_;
   m_trait = Expression::e_trait_inheritance;
}

ExprInherit::ExprInherit( const Symbol* symbol, int line, int chr ):
   ExprVector( line, chr ),
   m_base(0),
   m_symbol( symbol )
{
   FALCON_DECLARE_SYN_CLASS( expr_inherit )
   symbol->incref();
   apply = apply_;
   m_trait = Expression::e_trait_inheritance;
}

ExprInherit::ExprInherit( Class* base, int line, int chr ):
   ExprVector( line, chr ),
   m_base( base ),
   m_symbol( Engine::getSymbol(base->name()) )
{
   FALCON_DECLARE_SYN_CLASS( expr_inherit )
   apply = apply_;
   m_trait = Expression::e_trait_inheritance;
}
   
ExprInherit::ExprInherit( const ExprInherit& other ):
   ExprVector( other ),
   m_base( other.m_base ),
   m_symbol( other.m_symbol )
{
   apply = apply_;
   if(m_symbol != 0 )
   {
      m_symbol->incref();
   }
   m_trait = Expression::e_trait_inheritance;
}
  
ExprInherit::~ExprInherit()
{
   if(m_symbol != 0 )
   {
      m_symbol->decref();
   }
}

void ExprInherit::base( Class* cls )
{
   m_base = cls;
   if( m_symbol == 0 )
   {
      m_symbol = Engine::getSymbol(cls->name());
   }
}


void ExprInherit::render( TextWriter* tw, int depth ) const
{
   tw->write(renderPrefix(depth));

   tw->write( m_symbol == 0 ? "?" : m_symbol->name() );
   if( ! _p->m_exprs.empty() )
   {
      tw->write( "(" );

      TreeStepVector_Private::ExprVector::const_iterator iter = _p->m_exprs.begin();
      while( _p->m_exprs.end() != iter )
      {
         TreeStep* param = *iter;
         if( _p->m_exprs.begin() != iter )
         {
            tw->write(", ");
         }
         // keep same depth
         param->render( tw, relativeDepth(depth) );
         ++iter;
      }

      tw->write( ")" );
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}


void ExprInherit::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprInherit* self = static_cast<const ExprInherit*>(ps);
   fassert( self->m_base != 0 );
   
   // we need to "produce" the parameters, were any of it.
   CodeFrame& cf = ctx->currentCode();
   int& seqId = cf.m_seqId;
   const TreeStepVector_Private::ExprVector& exprs = self->_p->m_exprs;
   int size = (int) exprs.size();   

   TRACE1("ExprInherit::apply_ with %d/%d parameters (depth: %d)", seqId, size, (int) ctx->dataSize() );

   while( seqId < size )
   {
      TRACE1("ExprInherit::apply_ looping %d/%d", seqId, size );

      TreeStep* exp = exprs[seqId++];
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
   fassert( iinst->isUser() );
   fassert( iinst->asClass() == self->m_base );
   
   // invoke the init operator directly
   if( self->m_base->op_init( ctx, iinst->asInst(), size ) )
   {
      // It's deep.
      return;
   }
   // we're in charge of popping the parameters.
   ctx->popData(size);
}

}

/* end of exprinherit.cpp */
