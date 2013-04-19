/*
   FALCON - The Falcon Programming Language.
   FILE: exprsummon.cpp

   Syntactic tree item definitions -- expression elements -- summon
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 18 Apr 2013 15:38:58 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprsummon.cpp"

#include <falcon/symbol.h>
#include <falcon/vmcontext.h>
#include <falcon/textwriter.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include <falcon/itemarray.h>

#include <falcon/errors/accesserror.h>
#include <falcon/errors/paramerror.h>

#include <falcon/psteps/exprsummon.h>
#include "exprvector_private.h"

namespace Falcon {

ExprSummonBase::ExprSummonBase( int line, int chr, bool isOptional ):
   ExprVector( line, chr ),
   m_bIsOptional(isOptional),
   m_stepSummoned(this)
{
   apply = apply_;
}


ExprSummonBase::ExprSummonBase( const String& message, int line, int chr, bool isOptional ):
   ExprVector( line, chr ),
   m_message( message ),
   m_bIsOptional(isOptional),
   m_stepSummoned(this)
{
   apply = apply_;
}


ExprSummonBase::ExprSummonBase( const ExprSummonBase& other ):
   ExprVector( other ),
   m_message( other.m_message ),
   m_bIsOptional(other.m_bIsOptional),
   m_stepSummoned(this)
{
   apply = apply_;
}


ExprSummonBase::~ExprSummonBase()
{
}


void ExprSummonBase::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   if( m_selector == 0 )
   {
      tw->write("/* Blank ExprSummonBase */");
   }
   else
   {
      m_selector->render( tw, relativeDepth(depth) );
      tw->write( m_bIsOptional ? ":?" : "::" );
      tw->write( m_message );
      tw->write("[ ");
      // and generate all the expressions, in inverse order.
      for( unsigned int i = 0; i < _p->m_exprs.size(); ++i )
      {
         if ( i > 0 )
         {
            tw->write(", ");
         }
         _p->m_exprs[i]->render(tw, relativeDepth(depth) );
      }

      tw->write(" ]");
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}


TreeStep* ExprSummonBase::selector() const
{
   return m_selector;
}


bool ExprSummonBase::selector( TreeStep* ts )
{
   if( ts->setParent(this) )
   {
      dispose(m_selector);
      m_selector = ts;
      return true;
   }

   return false;
}



void ExprSummonBase::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprSummon* self = static_cast<const ExprSummon*>(ps);
   fassert( self->m_selector != 0 );

   CodeFrame& cf = ctx->currentCode();
   int32& seqId = cf.m_seqId;

   // where are we left?
   TRACE( "ExprSummon::apply_ on %s at step %d", self->m_selector->describe().c_ize(), seqId );

   const String* msg = &self->m_message;
   switch( seqId )
   {
   case 0:
      ++seqId;
      // check special messages.
      if( *msg == "respondTo" )
      {
         // generate respond request -- we won't be back.
         ctx->resetCode(&self->m_stepResponded);
         Class* cls = 0;
         void* data = 0;
         ctx->topData().forceClassInst(cls, data);
         cls->op_respondTo(ctx, data, self->m_message);
         return;
      }
      else if( *msg == "summon" || *msg == "vsummon")
      {
         //The message is the first parameter...
         TreeStep* first = self->nth(0);
         if( first == 0 )
         {
            throw FALCON_SIGN_XERROR( ParamError, e_param_arity, .extra("::summon S") );
         }
         ctx->resetCode(&self->m_stepSummoned);
         ctx->pushCode(first);
         return;
      }
      /* no break */

   // first, generate the selector expression.
   case 1:
      ++seqId;
      if( ctx->stepInYield(ps) ) {
         return;
      }
      /* no break */

   // then, check if the expression responds to the messages.
   case 2:
      {
         ++seqId;
         Class* cls = 0;
         void* data = 0;
         ctx->topData().forceClassInst(cls, data);
         cls->op_respondTo(ctx, data, self->m_message);

         if( &cf != &ctx->currentCode() ){
            return;
         }
      }
      /* no break */

   // Check the summoning
   case 3:

      bool responds = ctx->topData().asBoolean();
      ctx->popData(); // op_respondTo added a value in the stack.
      if( ! responds )
      {
         if( self->m_bIsOptional )
         {
            ctx->popCode(); // we're done
            ctx->popData(); // remove also the item we pushed
            return;
         }

         // not responding to a mandatory message is an error.
         throw FALCON_SIGN_XERROR(AccessError, e_not_responding, .extra(self->m_message) );
      }

      // generate all the expressions
      {
         int32 n = self->arity();
         if( n > 0 )
         {
            seqId++;
            // push last to first to generate first-to-last
            while( n > 0 )
            {
               --n;
               ctx->pushCode( self->nth(n) );
            }
            return;
         }
      }

      /* no break */
   }

   // we're done.
   ctx->popCode();

   // perform the call to summon
   int32 arity = self->arity();
   Item& target = ctx->opcodeParam(arity);
   Class* cls = 0;
   void* data = 0;
   target.forceClassInst(cls, data);
   cls->op_summon(ctx, data, self->m_message, arity);
}


//=====================================================================
// Summon specialization
//

ExprSummon::ExprSummon( int line, int chr ):
    ExprSummonBase( line, chr, false )
{
   FALCON_DECLARE_SYN_CLASS( expr_summon );
}

ExprSummon::ExprSummon( const String& name,  int line, int chr ):
         ExprSummonBase( name, line, chr, false )
{
   FALCON_DECLARE_SYN_CLASS( expr_summon );
}

ExprSummon::ExprSummon( const ExprSummon& other ):
   ExprSummonBase( other )
{
}

ExprSummon::~ExprSummon()
{}


//=====================================================================
// OptSummon specialization
//

ExprOptSummon::ExprOptSummon( int line, int chr ):
    ExprSummonBase( line, chr, true )
{
   FALCON_DECLARE_SYN_CLASS( expr_optsummon );
}

ExprOptSummon::ExprOptSummon( const String& name,  int line, int chr ):
         ExprSummonBase( name, line, chr, true )
{
   FALCON_DECLARE_SYN_CLASS( expr_optsummon );
}

ExprOptSummon::ExprOptSummon( const ExprSummon& other ):
   ExprSummonBase( other )
{
}

ExprOptSummon::~ExprOptSummon()
{}
   
}

/* end of exprsym.cpp */
