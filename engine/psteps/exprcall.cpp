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
#include <falcon/textwriter.h>

#include <falcon/psteps/exprcall.h>
#include <falcon/psteps/exprtree.h>
#include <falcon/psteps/exprnamed.h>

#include <falcon/synclasses.h>
#include <falcon/synclasses_id.h>
#include <falcon/engine.h>

#include <vector>

#include "exprvector_private.h"

namespace Falcon {

ExprCall::ExprCall( int line, int chr ):
   ExprVector( line, chr ),
   m_callExpr(0),
   m_bHasNamedParams(false),
   m_stepEvalPosParams(this),
   m_stepEvalEtaParams(this),
   m_stepEvalNamedParams(this),
   m_stepInvoke(this)
{
   FALCON_DECLARE_SYN_CLASS( expr_call )      
   apply = apply_;
}


ExprCall::ExprCall( Expression* callee, int line, int chr ):
   ExprVector( line, chr ),
   m_callExpr(callee),
   m_bHasNamedParams(false),
   m_stepEvalPosParams(this),
   m_stepEvalEtaParams(this),
   m_stepEvalNamedParams(this),
   m_stepInvoke(this)
{
   FALCON_DECLARE_SYN_CLASS( expr_call )
   apply = apply_;
}


ExprCall::ExprCall( const ExprCall& other ):
   ExprVector( other ),
   m_bHasNamedParams(other.m_bHasNamedParams),
   m_stepEvalPosParams(this),
   m_stepEvalEtaParams(this),
   m_stepEvalNamedParams(this),
   m_stepInvoke(this)
{
   FALCON_DECLARE_SYN_CLASS( expr_call )
   if( other.m_callExpr != 0 ) {
      m_callExpr = other.m_callExpr->clone();
      m_callExpr->setParent(this);
   }
   else {
      m_callExpr = 0;
   }
   apply = apply_;
}


ExprCall::~ExprCall()
{
   dispose( m_callExpr );
}


bool ExprCall::simplify( Item& ) const
{
   return false;
}


TreeStep* ExprCall::selector() const
{
   return m_callExpr;
}


bool ExprCall::selector( TreeStep* e )
{
   if( e->setParent(this))
   {
      dispose( m_callExpr );
      m_callExpr = e;
      return true;
   }
   return false;
}


bool  ExprCall::insert( int32 pos, TreeStep* element )
{
   if( ExprVector::insert(pos, element) )
   {
      checkPositionalParameter( element );
      return true;
   }

   return false;
}


bool ExprCall::setNth( int32 n, TreeStep* ts )
{
   if( ExprVector::setNth(n, ts) )
   {
      checkPositionalParameter( ts );
      return true;
   }

   return false;
}


bool ExprCall::append( TreeStep* element )
{
   if( ExprVector::append(element) )
   {
      checkPositionalParameter( element );
      return true;
   }

   return false;
}


void ExprCall::checkPositionalParameter( TreeStep* element )
{
   if( element->category() == TreeStep::e_cat_expression  )
   {
      Expression* expr = static_cast<Expression*>(element);
      if( expr->trait() == Expression::e_trait_named )
      {
         m_bHasNamedParams = true;
      }
   }
}

void ExprCall::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   if( m_callExpr == 0 )
   {
      tw->write("/* Blank ExprCall */");
   }
   else
   {
      m_callExpr->render( tw, relativeDepth(depth) );
      tw->write("( ");
      // and generate all the expressions, in inverse order.
      for( unsigned int i = 0; i < _p->m_exprs.size(); ++i )
      {
         if ( i > 0 )
         {
            tw->write(", ");
         }
         _p->m_exprs[i]->render(tw, relativeDepth(depth) );
      }

      tw->write(" )");
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}

inline bool isEta( const Item& top, int& pcount )
{
  Class* cls = 0;
  void* vts = 0;
  top.forceClassInst(cls, vts);

  switch(cls->typeID())
  {
     case FLC_CLASS_ID_FUNC:
     {
        Function* f = top.asFunction();
        pcount = f->paramCount();
        return f->isEta();
     }
     break;

     case FLC_CLASS_ID_CLASS:
     {
        Class* cls = static_cast<Class*>(top.asInst());
        Function* ctr = cls->getConstructor();
        if( ctr != 0 )
        {
           pcount = ctr->paramCount();
           return ctr->isEta();
        }
     }
     break;

     case FLC_ITEM_METHOD:
     {
        Function* f = top.asMethodFunction();
        pcount = f->paramCount();
        return f->isEta();
     }
     break;

     case FLC_CLASS_ID_CLOSURE:
     {
        Closure* cl = static_cast<Closure*>(top.asInst());
        pcount = cl->closed()->paramCount();
        return cl->closed()->isEta();
     }
     break;

     case FLC_CLASS_ID_TREESTEP:
     {
        if( cls->userFlags() == FALCON_SYNCLASS_ID_TREE) {
           pcount = static_cast<ExprTree*>(vts)->paramCount();
           return static_cast<ExprTree*>(vts)->isEta();
        }
     }
     break;
  }

  return false;
}


inline void invoke( VMContext* ctx, int pcount )
{
   register Item& top = *(&ctx->topData()-pcount);

   switch(top.type())
   {
      case FLC_CLASS_ID_FUNC:
         {
            // this is just a shortcut for a very common case.
            Function* f = top.asFunction();
            ctx->callInternal( f, pcount );
         }
         break;

      case FLC_CLASS_ID_CLOSURE:
         {
            // this is just a shortcut for a very common case.
            Closure* f = static_cast<Closure*>(top.asInst());
            ctx->callInternal( f, pcount );
         }
         break;

      case FLC_ITEM_METHOD:
         {
            Item old = top;
            Function* f = top.asMethodFunction();
            old.unmethodize();
            ctx->callInternal( f, pcount, old );
         }
         break;

      default:
         {
            Class* cls = 0;
            void* inst = 0;
            top.forceClassInst( cls, inst );
            cls->op_call( ctx, pcount, inst );
         }
         break;
   }
}


inline int findParameter( const Item& called, const String& pname )
{
   const SymbolMap* symbols = 0;
   switch(called.type())
   {
      case FLC_CLASS_ID_FUNC:
         symbols = &called.asFunction()->parameters();
         break;

      case FLC_CLASS_ID_CLOSURE:
         symbols = &static_cast<Closure*>(called.asInst())->closed()->parameters();
         break;

      case FLC_ITEM_METHOD:
         symbols = &called.asMethodFunction()->parameters();
         break;

      case FLC_CLASS_ID_CLASS:
         {
            Class* cls = static_cast<Class*>(called.asInst());
            if( cls->getConstructor() != 0 )
            {
               symbols = &cls->getConstructor()->parameters();
            }
         }
         break;

      case FLC_CLASS_ID_TREESTEP:
           if( called.asClass()->userFlags() == FALCON_SYNCLASS_ID_TREE) {
              symbols = static_cast<ExprTree*>(called.asInst())->parameters();
           }
           break;
   }

   if( symbols == 0 )
   {
      throw FALCON_SIGN_ERROR( ParamError, e_param_noname );
   }

   int pos = symbols->find(pname);
   if( pos == -1 )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_param_notfound, .extra(pname) );
   }

   return pos;
}

void ExprCall::resolveUnquote( VMContext* ctx, const UnquoteResolver& res )
{
   ExprVector::resolveUnquote(ctx, res);

   class ElemUR: public UnquoteResolver
   {
   public:
      ElemUR( ExprCall* cs ):
         m_parent(cs)
      {}

      virtual ~ElemUR() {}

      void onUnquoteResolved( TreeStep* newStep ) const
      {
         m_parent->selector(newStep);
      }

      ExprCall* m_parent;
   };

   ElemUR ur(this);
   if( m_callExpr != 0 )
   {
      m_callExpr->resolveUnquote(ctx, ur);
   }
}

//=================================================================
// Apply
//

void ExprCall::apply_( const PStep* v, VMContext* ctx )
{
   const ExprCall* self = static_cast<const ExprCall*>(v);
   TRACE2( "Apply CALL %s", self->describe().c_ize() );

   fassert( self->m_callExpr != 0 );
   
   // prepare the call expression.
   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
      // Generate the called item.
      case 0:
         cf.m_seqId = 1;
         if( ctx->stepInYield( self->m_callExpr, cf ) )
         {
            return;
         }
         /* no break */
   }     
   
   // Evaluate the eta-ness of the called item
   // no need to increase seqID, we won't be here with seqId=1 anymore
   ctx->popCode();

   int pcount = self->_p->m_exprs.size();
   if (pcount > 0)
   {
      int declPCount = 0;
      if( isEta( ctx->topData(), declPCount ) )
      {
         ctx->stepIn(&self->m_stepEvalEtaParams);
      }
      else if( self->hasNamedParams() )
      {
         ctx->pushCode(&self->m_stepEvalNamedParams);
         // prepare the data stack
         int depth = declPCount > pcount ? declPCount : pcount;
         ctx->addLocals( depth );     // enough space for the parameters
         ctx->pushData((int64)depth); // max depth
         ctx->pushData((int64)0);     // current depth

         // generate first expression
         ctx->pushCode(self->_p->m_exprs.front());
      }
      else {
         ctx->stepIn(&self->m_stepEvalPosParams);
      }
   }
   else {
      ctx->stepIn(&self->m_stepInvoke);
   }
}


void ExprCall::PStepEvalEtaParams::apply_( const PStep* ps, VMContext* ctx )
{
   const PStepEvalEtaParams* pself = static_cast<const PStepEvalEtaParams*>(ps);
   const ExprCall* self = pself->m_owner;

   TRACE2( "Apply ExprCall::PStepEvalEtaParams %s", self->describe().c_ize() );

   TreeStepVector_Private::ExprVector::iterator pos = self->_p->m_exprs.begin();
   TreeStepVector_Private::ExprVector::iterator end = self->_p->m_exprs.end();

   while( pos < end )
   {
      TreeStep* expr = *pos;
      ctx->pushData( Item(expr->handler(), expr ) );
      ++pos;
   }

   ctx->popCode();
   ctx->stepIn(&self->m_stepInvoke);
}


void ExprCall::PStepEvalNamedParams::apply_( const PStep* ps, VMContext* ctx )
{
   /*
    * Stack is:
    * TOP <generated expression>
    *     <current max depth>
    *     <depth>
    * .
    * .   <depth items>
    * .
    *     <called item>
    */

   const PStepEvalNamedParams* pself = static_cast<const PStepEvalNamedParams*>(ps);
   const ExprCall* self = pself->m_owner;

   int curDepth = (int) ctx->opcodeParam(1).asInteger();
   int depth = (int) ctx->opcodeParam(2).asInteger();
   int pcount = self->_p->m_exprs.size();

   CodeFrame& cf = ctx->currentCode();
   TRACE2( "Apply ExprCall::PStepEvalNamedParams %d/%d %s ",
            cf.m_seqId, pcount,  self->describe().c_ize() );

   // the result is at top; we must move it down to the correct position
   Item* base = ctx->opcodeParams(depth + 3);

   TreeStepVector_Private::ExprVector::iterator pos = self->_p->m_exprs.begin() + cf.m_seqId;
   TreeStep* ts = *pos;
   // a named expression?
   if( ts->category() == TreeStep::e_cat_expression && static_cast<Expression*>(ts)->trait() == Expression::e_trait_named )
   {
      ExprNamed* named = static_cast<ExprNamed*>(ts);
      // will throw if the parameter is not found
      int place = findParameter( *(base-1), named->name() );
      base[place] = ctx->topData();

      // mark that we have found a non-positional parameter
      ctx->opcodeParam(2).setOob(true);

      // eventually, resize max depth
      if( place >= curDepth )
      {
         curDepth = place+1;
      }
   }
   else {
      // No positional paramter after named parameters
      if( ctx->opcodeParam(2).isOob() )
      {
         throw FALCON_SIGN_ERROR( ParamError, e_param_compo );
      }
      base[curDepth] = ctx->topData();
      curDepth++;
   }

   cf.m_seqId++;
   if( cf.m_seqId == pcount )
   {
      //we're done
      ctx->popCode();
      ctx->popData(3 + (depth - curDepth) ); // top and other 2 local vars
      invoke(ctx, curDepth);
   }
   else {
      // generate next
      ctx->popData();
      ctx->topData().setInteger(curDepth);
      ++pos;
      ctx->pushCode( *pos );
   }

}


void ExprCall::PStepEvalPosParams::apply_( const PStep* ps, VMContext* ctx )
{
   const PStepEvalPosParams* pself = static_cast<const PStepEvalPosParams*>(ps);
   const ExprCall* self = pself->m_owner;

   TRACE2( "Apply ExprCall::PStepEvalPosParams %s", self->describe().c_ize() );

   CodeFrame& cf = ctx->currentCode();

   TreeStepVector_Private::ExprVector::iterator pos = self->_p->m_exprs.begin() + cf.m_seqId;
   TreeStepVector_Private::ExprVector::iterator end = self->_p->m_exprs.end();

   while( pos < end )
   {
      cf.m_seqId++;
      if( ctx->stepInYield( *pos, cf ) )
      {
         return;
      }
      ++pos;
   }

   ctx->popCode();
   ctx->stepIn(&self->m_stepInvoke);
}


void ExprCall::PStepInvoke::apply_( const PStep* ps, VMContext* ctx )
{
   const PStepInvoke* pself = static_cast<const PStepInvoke*>(ps);
   const ExprCall* self = pself->m_owner;

   TRACE2( "Apply ExprCall::PStepInvoke %s", self->describe().c_ize() );

   // we're out of business
   ctx->popCode();

   // now, top points to our function value.
   int pcount = self->_p->m_exprs.size();
   ctx->callerLine( self->sr().line() );
   invoke( ctx, pcount );
}

}

/* end of exprcall.cpp */
