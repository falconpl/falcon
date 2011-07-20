/*
   FALCON - The Falcon Programming Language.
   FILE: exprcall.cpp

   Expression controlling item (function) call
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 21:19:26 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/exprcall.h>
#include <falcon/pcode.h>
#include <falcon/pseudofunc.h>
#include <falcon/trace.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>

#include <vector>

namespace Falcon {

class ExprCall::Private {
public:
   std::vector<Expression*> m_params;
};

ExprCall::ExprCall( Expression* op1 ):
   Expression( t_funcall ),
   m_func(0),
   m_callExpr(op1)
{
   _p = new Private;
   apply = apply_;
}


ExprCall::ExprCall( PseudoFunction* f ):
   Expression( t_funcall ),
   m_func(f),
   m_callExpr(0)
{
   _p = new Private;
   apply = apply_dummy_;
}

ExprCall::ExprCall( const ExprCall& other ):
   Expression( other )
{
   apply = other.apply;
   m_func = other.m_func;
   m_callExpr = other.m_callExpr;

   _p = new Private;
   _p->m_params.reserve( other._p->m_params.size() );
   std::vector<Expression*>::const_iterator iter = other._p->m_params.begin();
   while(iter != other._p->m_params.end())
   {
      _p->m_params.push_back((*iter)->clone());
      ++iter;
   }
}

ExprCall::~ExprCall()
{
   // and generate all the expressions, in inverse order.
   for( unsigned int i = 0; i < _p->m_params.size(); ++i )
   {
      delete _p->m_params[i];
   }

   delete _p;
}


/** Function call. */
void ExprCall::precompile( PCode* pcode ) const
{
   TRACE3( "Precompiling call: %p (%s)", pcode, describe().c_ize() );

   // first, precompile the called object, if any,
   if( m_callExpr != 0 )
   {
      m_callExpr->precompile( pcode );
   }

   // precompile all parameters in order.
   for( uint32 i = 0; i < _p->m_params.size(); ++i )
   {
      _p->m_params[i]->precompile( pcode );
   }

   // and finally push the step to do the call.
   if( m_callExpr != 0 )
   {
      pcode->pushStep( this );
   }
   else
   {
      // pseudofunctions can be pushed directly.
      pcode->pushStep( m_func->pstep() );
   }
}


bool ExprCall::simplify( Item& ) const
{
   return false;
}


void ExprCall::apply_dummy_( const PStep* DEBUG_ONLY(v), VMContext* )
{
   TRACE2( "Apply CALL -- dummy! %s", static_cast<const ExprCall*>(v)->describe().c_ize());
   // we should never be called
}


void ExprCall::apply_( const PStep* v, VMContext* ctx )
{
   static Engine* eng = Engine::instance();
   const ExprCall* self = static_cast<const ExprCall*>(v);
   TRACE2( "Apply CALL %s", self->describe().c_ize() );

   int pcount = self->_p->m_params.size();
   register Item& top = *(&ctx->topData()-pcount);

   switch(top.type())
   {
      case FLC_ITEM_FUNC:
         {
            Function* f = top.asFunction();
            ctx->call( f, pcount, true );
         }
         break;

      case FLC_ITEM_METHOD:
         {
            Item old = top;
            Function* f = top.asMethodFunction();
            old.unmethodize();
            ctx->call( f, pcount, old );
         }
         break;

      // TODO: Class Method

      case FLC_ITEM_USER:
         {
            Class* cls = top.asClass();
            void* inst = top.asInst();
            cls->op_call( ctx, pcount, inst );
         }
         break;

      default:
         {
            Class* cls = eng->getTypeClass( top.type() );
            cls->op_call( ctx, pcount, 0 );
         }
   }

}


ExprCall& ExprCall::addParam( Expression* p )
{
   _p->m_params.push_back( p );
   return *this;
}


Expression* ExprCall::getParam( int n ) const
{
   return _p->m_params[ n ];
}

int ExprCall::paramCount() const
{
   return _p->m_params.size();
}

void ExprCall::describe( String& ret ) const
{
   String params;
   // and generate all the expressions, in inverse order.
   for( unsigned int i = 0; i < _p->m_params.size(); ++i )
   {
      if ( params.size() )
      {
         params += ", ";
      }
      params += _p->m_params[i]->describe();
   }

   if( m_callExpr != 0 )
   {
      ret = m_callExpr->describe() + "(" + params +  ")";
   }
   else
   {
      ret = m_func->name() + "(" + params +  ")";
   }
}

}

/* end of exprcall.cpp */
