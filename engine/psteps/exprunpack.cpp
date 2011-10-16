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
#include <falcon/expression.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/pcode.h>
#include <falcon/operanderror.h>
#include <falcon/codeerror.h>
#include <falcon/itemarray.h>
#include <falcon/symbol.h>

namespace Falcon {

class ExprUnpack::Private {
public:
   std::vector<Symbol*> m_params;
};

//=========================================================
// Unpack

ExprUnpack::ExprUnpack( Expression* op1, bool isTop ):
   Expression(t_unpack),
   m_expander(op1),
   m_bIsTop( isTop ),
   _p( new Private )
{
   apply = apply_;
}

ExprUnpack::ExprUnpack( const ExprUnpack& other ):
         Expression(other)
{
   _p = new Private;
   
   m_expander = other.m_expander->clone();

   _p->m_params.reserve(other._p->m_params.size());
   std::vector<Symbol*>::const_iterator iter = other._p->m_params.begin();
   while( iter != other._p->m_params.end() )
   {
      _p->m_params.push_back( *iter );
      ++iter;
   }
}

ExprUnpack::~ExprUnpack()
{
   delete m_expander;
}

bool ExprUnpack::simplify( Item& ) const
{
   return false;
}

void ExprUnpack::describeTo( String& ret ) const
{
   String params;
   // and generate all the expressions, in inverse order.
   for( unsigned int i = 0; i < _p->m_params.size(); ++i )
   {
      if ( params.size() )
      {
         params += ", ";
      }
      params += _p->m_params[i]->name();
   }

   ret = params + " = " + m_expander->describe();
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
   return *this;
}


void ExprUnpack::precompile( PCode* pcode ) const
{
   TRACE3( "Precompiling unpack: %p (%s)", pcode, describe().c_ize() );

   // first, precompile the
   m_expander->precompile( pcode );

   pcode->pushStep( this );
}


void ExprUnpack::apply_( const PStep* ps, VMContext* ctx )
{
   TRACE3( "Apply unpack: %p (%s)", ps, ps->describe().c_ize() );

   ExprUnpack* self = (ExprUnpack*) ps;
   std::vector<Symbol*> &syms = self->_p->m_params;
   size_t pcount = syms.size();
   
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
      *syms[i]->value() = array[i];
   }

   // leave the expander in the stack.
}

}

/* end of exprunpack.cpp */
