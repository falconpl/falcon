/*
   FALCON - The Falcon Programming Language.
   FILE: exprindex.cpp

   Syntactic tree item definitions -- Index accessor
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Bgin: Sat, 30 Jul 2011 16:26:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprindex.cpp"

#include <falcon/vmcontext.h>
#include <falcon/trace.h>
#include <falcon/stdsteps.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <falcon/psteps/exprindex.h>

#include <falcon/psteps/exprsym.h>
#include <falcon/symbol.h>
#include <falcon/psteps/exprvalue.h>

namespace Falcon
{

bool ExprIndex::simplify( Item& ) const
{
   //ToDo possibly add simplification for indexing.
   return false;
}

class Activity_GetIndex {
public:
   inline static void operate( Class* cls, VMContext* ctx, void* instance ) {
      cls->op_getIndex( ctx, instance );
   }
   inline static const char* mode() { return ""; }
};

class Activity_SetIndex {
public:
   inline static void operate( Class* cls, VMContext* ctx, void* instance ) {
      cls->op_setIndex( ctx, instance );
   }
   inline static const char* mode() { return "lvalue"; }
};
   
   

template< class activity>
inline void generic_apply( const ExprIndex* self, VMContext* ctx )
{   
   TRACE2( "Apply %s \"%s\"", activity::mode(), self->describe().c_ize() );
   
   fassert( self->first() != 0 );
   fassert( self->second() != 0 );
   
   CodeFrame& cf = ctx->currentCode();
   Expression::t_trait trait;
   register Expression* current;
   
   switch( cf.m_seqId )
   {
      // first time around
      case 0:
         cf.m_seqId = 1;
         // don't bother to call if we know what it is
         // TODO: it's better optimized through a specific pstep.
         current = self->first();
         trait = current->trait();
         if( trait == Expression::e_trait_symbol )
         {
            ctx->pushData(*static_cast<ExprSymbol*>( current )->symbol()->value(ctx));
         }
         else {
            if( ctx->stepInYield( current, cf ) ) return;
         }
         // fallthrough
         
      case 1:
         cf.m_seqId = 2;
         // don't bother to call if we know what it is
         // TODO: it's better optimized through a specific pstep.
         current = self->second();
         trait = current->trait();
         if( trait == Expression::e_trait_value )
         {
            ctx->pushData(static_cast<ExprValue*>( current )->item());
         }
         else {
            if( ctx->stepInYield( current, cf ) ) return;
         }
         // fallthrough
   }
   
   // we're done here.
   ctx->popCode();
   
   // now apply the index.
   Class* cls;
   void* instance;
   
   //acquire the class
   (&ctx->topData()-1)->forceClassInst(cls, instance);
   // apply the set or get index
   activity::operate( cls, ctx, instance );   
}

void ExprIndex::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprIndex* self = static_cast<const ExprIndex*>(ps);
   generic_apply<Activity_GetIndex>( self, ctx );
}


void ExprIndex::PstepLValue::apply_( const PStep* DEBUG_ONLY(ps), VMContext* ctx )
{
   const ExprIndex* self = static_cast<const ExprIndex::PstepLValue*>(ps)->m_owner;
   generic_apply<Activity_SetIndex>( self, ctx );
}


void ExprIndex::describeTo( String& ret, int depth ) const
{
   if( m_first == 0 || m_second == 0 )
   {
      ret = "<Blank ExprIndex>";
      return;
   }
   
   ret = "(" + m_first->describe(depth+1) + "[" + m_second->describe(depth+1) + "])";
}


}

/* end of exprindex.cpp */
