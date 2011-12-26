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

#include <falcon/pseudofunc.h>
#include <falcon/trace.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>

#include <falcon/psteps/exprcall.h>

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
   apply = apply_;
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


bool ExprCall::simplify( Item& ) const
{
   return false;
}

void ExprCall::apply_( const PStep* v, VMContext* ctx )
{
   static Engine* eng = Engine::instance();
   const ExprCall* self = static_cast<const ExprCall*>(v);
   TRACE2( "Apply CALL %s", self->describe().c_ize() );
   int pcount = self->_p->m_params.size();

   // prepare the call expression.
   CodeFrame& cf = ctx->currentCode();
   if( cf.m_seqId == 0 )  
   {
      // got to compile or push the call item.
      cf.m_seqId = 1;
      if( self->m_callExpr != 0 )
      {
         if( ctx->stepInYield( self->m_callExpr, cf ) )
         {
            return;
         }
      }
      else 
      {
         fassert( self->m_func != 0 );
         // can we call directly our nice function?
         // call directly our pseudofunction?
         if( self->m_func->paramCount() == pcount )
         {
            ctx->resetCode( self->m_func->pstep() );
            return;
         }
         
         // Otherwise, we must handle this as a normal function
         // -- but notice that the compiler should have blocked us.         
         ctx->pushData( self->m_func );
      }
   }
   
   // now got to generate all the paraeters, if any.
   // Notice that seqId here is nparam + 1, as 0 is for the function itself.
   
   if( pcount >= cf.m_seqId )
   {
      std::vector<Expression*>::iterator pos = self->_p->m_params.begin() + (cf.m_seqId-1);
      std::vector<Expression*>::iterator end = self->_p->m_params.end();
      while( pos < end )
      {
         cf.m_seqId++;
         if( ctx->stepInYield( *pos, cf ) )
         {
            return;
         }
         ++pos;
      }
   }   
   
   // anyhow, we're out of business.
   ctx->popCode();

   // now, top points to our function value.
   register Item& top = *(&ctx->topData()-pcount);

   switch(top.type())
   {
      case FLC_ITEM_FUNC:
         {
            Function* f = top.asFunction();
            ctx->call( f, pcount );
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

void ExprCall::describeTo( String& ret, int depth ) const
{
   String params;
   // and generate all the expressions, in inverse order.
   for( unsigned int i = 0; i < _p->m_params.size(); ++i )
   {
      if ( params.size() )
      {
         params += ", ";
      }
      params += _p->m_params[i]->describe(depth+1);
   }

   if( m_callExpr != 0 )
   {
      ret = m_callExpr->describe(depth+1) + "(" + params +  ")";
   }
   else
   {
      ret = m_func->name() + "(" + params +  ")";
   }
}

}

/* end of exprcall.cpp */
