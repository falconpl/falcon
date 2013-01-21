/*
   FALCON - The Falcon Programming Language.
   FILE: exprunpack.cpp

   Syntactic tree item definitions -- expression elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 13:39:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/psteps/exprunpack.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/errors/operanderror.h>
#include <falcon/errors/codeerror.h>
#include <falcon/itemarray.h>
#include <falcon/symbol.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <vector>

namespace Falcon {


class ExprUnpack::Private {
public:
   std::vector<Symbol*> m_params;
};

//=========================================================
// Unpack

ExprUnpack::ExprUnpack( Expression* op1, bool isTop, int line, int chr ):
   Expression(line, chr),
   m_expander(op1),
   m_bIsTop( isTop ),
   _p( new Private )
{
   FALCON_DECLARE_SYN_CLASS( expr_unpack )
   apply = apply_;
   m_trait = Expression::e_trait_composite;
}

ExprUnpack::ExprUnpack( int line, int chr ):
   Expression(line, chr),
   m_expander(0),
   m_bIsTop( false ),
   _p( new Private )
{
   FALCON_DECLARE_SYN_CLASS( expr_unpack )
   apply = apply_;
   m_trait = Expression::e_trait_composite;
}

ExprUnpack::ExprUnpack( const ExprUnpack& other ):
   Expression(other)
{
   _p = new Private;
   m_trait = Expression::e_trait_composite;
   m_expander = other.m_expander->clone();

   _p->m_params.reserve(other._p->m_params.size());
   std::vector<Symbol*>::const_iterator iter = other._p->m_params.begin();
   while( iter != other._p->m_params.end() )
   {
      (*iter)->incref();
      _p->m_params.push_back( *iter );
      ++iter;
   }
}

ExprUnpack::~ExprUnpack()
{
   delete m_expander;
   std::vector<Symbol*>::const_iterator iter = _p->m_params.begin();
   while( iter != _p->m_params.end() )
   {
      (*iter)->decref();
      ++iter;
   }
}

bool ExprUnpack::simplify( Item& ) const
{
   return false;
}

void ExprUnpack::describeTo( String& ret, int depth ) const
{
   String params;
   // and generate all the expressions, in inverse order.
   for( unsigned int i = 0; i < _p->m_params.size(); ++i )
   {
      if ( i > 0 )
      {
         params += ", ";
      }
      params += _p->m_params[i]->name();
   }

   ret = params + " = " + m_expander->describe(depth+1);
}


int ExprUnpack::targetCount() const
{
   return _p->m_params.size();
}

Symbol* ExprUnpack::getAssignand( int i) const
{
   return _p->m_params[i];
}

ExprUnpack& ExprUnpack::addAssignand(Symbol* e)
{
   _p->m_params.push_back(e);
   e->incref();
   return *this;
}


void ExprUnpack::apply_( const PStep* ps, VMContext* ctx )
{
   TRACE3( "Apply unpack: %p (%s)", ps, ps->describe().c_ize() );

   const ExprUnpack* self = static_cast<const ExprUnpack*>(ps);
   std::vector<Symbol*> &syms = self->_p->m_params;
   size_t pcount = syms.size();
   
   // eventually generate the expander.
   CodeFrame& cf = ctx->currentCode();
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_expander, cf ) )
      {
         return;
      }
   }
   
   // we won't be called anymore
   ctx->popCode();
   register Item& expander = ctx->topData();
   if ( ! expander.isArray() )
   {
      // no need to throw, we're going to get back in the VM.
      throw
         new OperandError( ErrorParam(e_unpack_size, __LINE__ ).extra("Not an array") );
   }
   ItemArray& array = *(ItemArray*) expander.asInst();

   if( pcount != array.length() )
   {
      throw
         new OperandError( ErrorParam(e_unpack_size, __LINE__ ).extra("Different size") );
   }

   size_t i;
   for( i = 0; i < pcount; ++i )
   {
      ctx->resolveSymbol(syms[i], true)->assign(array[i]);
   }

   // leave the expander in the stack.
}

}

/* end of exprunpack.cpp */
