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
#include <falcon/textwriter.h>

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
   register TreeStep* current;

   switch( cf.m_seqId )
   {
      // first time around
      case 0:
         cf.m_seqId = 1;
         // don't bother to call if we know what it is
         // TODO: it's better optimized through a specific pstep.
         current = self->first();
         trait = current->category() == TreeStep::e_cat_expression ? static_cast<Expression*>(current)->trait() : Expression::e_trait_none;
         if( trait == Expression::e_trait_symbol )
         {
            Item* value = ctx->resolveSymbol(static_cast<ExprSymbol*>( current )->symbol(), false);
            ctx->pushData( *value );
         }
         else {
            if( ctx->stepInYield( current, cf ) ) return;
         }
         /* no break */

      case 1:
         cf.m_seqId = 2;
         // don't bother to call if we know what it is
         // TODO: it's better optimized through a specific pstep.
         current = self->second();
         trait = current->category() == TreeStep::e_cat_expression ? static_cast<Expression*>(current)->trait() : Expression::e_trait_none;
         if( trait == Expression::e_trait_value )
         {
            ctx->pushData(static_cast<ExprValue*>( current )->item());
         }
         else {
            if( ctx->stepInYield( current, cf ) ) return;
         }
         break;
   }


   // now apply the index.
   Class* cls;
   void* instance;

   //acquire the class
   (&ctx->topData()-1)->forceClassInst(cls, instance);

   // we're done here.
   // apply the set or get index
   FALCON_POPCODE_CONDITIONAL(ctx, self,
            activity::operate( cls, ctx, instance );
   );
}

void ExprIndex::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprIndex* self = static_cast<const ExprIndex*>(ps);
   generic_apply<Activity_GetIndex>( self, ctx );
}


void ExprIndex::PstepLValue::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprIndex* self = static_cast<const ExprIndex::PstepLValue*>(ps)->m_owner;
   generic_apply<Activity_SetIndex>( self, ctx );
}


void ExprIndex::render( TextWriter* tw, int depth ) const
{
   tw->write(renderPrefix(depth));

   if( m_first == 0 || m_second == 0 )
   {
      tw->write( "/* Blank ExprIndex */" );
   }
   else
   {
      m_first->render( tw, relativeDepth(depth) );
      tw->write("[");
      m_second->render( tw, relativeDepth(depth) );
      tw->write("]");
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}

const String& ExprIndex::exprName() const
{
   static String name("[]");
   return name;
}

}

/* end of exprindex.cpp */
