/*
   FALCON - The Falcon Programming Language.
   FILE: exprmultiunpack.cpp

   Syntactic tree item definitions -- expression elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 14:08:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/psteps/exprmultiunpack.h>
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

class ExprMultiUnpack::Private {
public:
   std::vector<Symbol*> m_params;
   std::vector<Expression*> m_assignee;
};

//=========================================================
// MultiUnpack
//
ExprMultiUnpack::ExprMultiUnpack( int line, int chr ):
   Expression(line, chr),
   m_bIsTop( true ),
   _p( new Private )
{
   FALCON_DECLARE_SYN_CLASS( expr_munpack )
   apply = apply_;
}

ExprMultiUnpack::ExprMultiUnpack( bool isTop, int line, int chr ):
   Expression(line, chr),
   m_bIsTop( isTop ),
   _p( new Private )
{
   FALCON_DECLARE_SYN_CLASS( expr_munpack )
   apply = apply_;
}


ExprMultiUnpack::ExprMultiUnpack( const ExprMultiUnpack& other ):
         Expression(other)
{
   apply = apply_;
    
   _p = new Private;

   _p->m_params.reserve(other._p->m_params.size());
   std::vector<Symbol*>::const_iterator iter = other._p->m_params.begin();
   while( iter != other._p->m_params.end() )
   {
      _p->m_params.push_back( *iter );
      ++iter;
   }

   _p->m_assignee.reserve(other._p->m_assignee.size());
   std::vector<Expression*>::const_iterator itere = other._p->m_assignee.begin();
   while( itere != other._p->m_assignee.end() )
   {
      _p->m_assignee.push_back( (*itere)->clone() );
      ++itere;
   }
}


ExprMultiUnpack::~ExprMultiUnpack()
{
   std::vector<Expression*>::const_iterator iter = _p->m_assignee.begin();
   while( iter != _p->m_assignee.end() )
   {
      delete *iter;
      ++iter;
   }
}


bool ExprMultiUnpack::simplify( Item& ) const
{
   return false;
}

void ExprMultiUnpack::describeTo( String& ret, int depth ) const
{
   String params;
   
   for( unsigned int i = 0; i < _p->m_params.size(); ++i )
   {
      if ( i >= 1 )
      {
         params += ", ";
      }
      params += _p->m_params[i]->name();
   }

   ret = params + " = ";

   for( unsigned int i = 0; i < _p->m_assignee.size(); ++i )
   {
      if ( i >= 1 )
      {
         params += ", ";
      }

      params += _p->m_assignee[i]->describe(depth+1);
   }
}


int ExprMultiUnpack::targetCount() const
{
   return _p->m_params.size();
}

Symbol* ExprMultiUnpack::getAssignand( int i) const
{
   return _p->m_params[i];
}

Expression* ExprMultiUnpack::getAssignee( int i) const
{
   return _p->m_assignee[i];
}


ExprMultiUnpack& ExprMultiUnpack::addAssignment( Symbol* e, Expression* expr)
{
   // save exprs and symbols in a parallel array
   _p->m_params.push_back(e);
   _p->m_assignee.push_back(expr);

   return *this;
}


void ExprMultiUnpack::apply_( const PStep* ps, VMContext* ctx )
{
   TRACE3( "Apply multi unpack: %p (%s)", ps, ps->describe().c_ize() );

   ExprMultiUnpack* self = (ExprMultiUnpack*) ps;
   std::vector<Symbol*> &syms = self->_p->m_params;

   // invoke all the expressions.
   CodeFrame& cf = ctx->currentCode();
   std::vector<Expression*>& assignee = self->_p->m_assignee;
   int size = (int) assignee.size();
   while( cf.m_seqId < size )
   {
      Expression* an = assignee[cf.m_seqId++];
      if( ctx->stepInYield( an, cf ) )
      {
         return;
      }
   }
   
   size_t pcount = syms.size();

   size_t i = 0;
   Item* topStack = &ctx->topData() - pcount+1;
   for( ; i < pcount; ++i, ++topStack )
   {
      *syms[i]->value( ctx ) = *topStack;
   }
   
   if ( self->isTop() )
   {
      // no need to create an array if noone is using it
      ctx->popData(pcount-1);
      ctx->topData().setNil();
   }
   else
   {
      ItemArray* retval = new ItemArray(pcount);
      i = 0;
      topStack = &ctx->topData() - pcount+1;
      for( ; i < pcount; ++i, ++topStack )
      {
         (*retval)[i] = *topStack;
      }
      ctx->popData(pcount-1);
      ctx->topData().setArray( retval );
   }
   
   ctx->popCode();
}

}

/* end of exprmultiunpack.cpp */
