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

#include <falcon/stderrors.h>

#include <falcon/psteps/exprsummon.h>
#include "exprvector_private.h"

namespace Falcon {

ExprSummonBase::ExprSummonBase( int line, int chr, bool isOptional ):
   ExprVector( line, chr ),
   m_selector(0),
   m_bIsOptional(isOptional)
{
   apply = apply_;
}


ExprSummonBase::ExprSummonBase( const String& message, int line, int chr, bool isOptional ):
   ExprVector( line, chr ),
   m_selector(0),
   m_message( message ),
   m_bIsOptional(isOptional)
{
   apply = apply_;
}


ExprSummonBase::ExprSummonBase( const ExprSummonBase& other ):
   ExprVector( other ),
   m_message( other.m_message ),
   m_bIsOptional(other.m_bIsOptional)
{
   apply = apply_;
   if( other.m_selector != 0 )
   {
      m_selector = other.m_selector->clone();
      m_selector->setParent(this);
   }
   else {
      m_selector = 0;
   }
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

   switch( seqId )
   {
   // first, generate the selector expression.
   case 0:
      ++seqId;
      if( ctx->stepInYield(self->selector()) ) {
         return;
      }
      /* no break */
   case 1:
     ++seqId;
     // generate all the parameters.
     {
        int32 count = self->arity();
        if ( count > 0 )
        {
           while (count > 0)
           {
              TreeStep* ts = self->nth(--count);
              ctx->pushCode(ts);
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
   cls->op_summon(ctx, data, self->m_message, arity, self->m_bIsOptional );
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
