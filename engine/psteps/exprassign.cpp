/*
   FALCON - The Falcon Programming Language.
   FILE: exprassign.cpp

   Syntactic tree item definitions -- Assignment operator.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/psteps/exprassign.h>
#include <falcon/vmcontext.h>
#include <falcon/textwriter.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>


namespace Falcon {

ExprAssign::ExprAssign(int line, int chr) :
   BinaryExpression( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_assign )
   apply = apply_;
   m_trait = e_trait_composite;
}

ExprAssign::ExprAssign( Expression* op1, Expression* op2, int line, int chr ):
   BinaryExpression( op1, op2, line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_assign )
   apply = apply_;
   m_trait = e_trait_composite;
}

ExprAssign::ExprAssign( const ExprAssign& other ):
   BinaryExpression( other )
{
   FALCON_DECLARE_SYN_CLASS( expr_assign )
   apply = apply_;
   m_trait = e_trait_composite;
}

bool ExprAssign::simplify( Item& ) const
{
   // TODO Simplify for closed symbols
   return false;
}


void ExprAssign::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   if( m_first == 0 || m_second == 0 )
   {
      tw->write("/* Blank ExprAssign */");
   }
   else
   {
      if( depth < 0 )
      {
         // it's a statement
         tw->write("( ");
      }

      m_first->render( tw, relativeDepth(depth) );
      tw->write( " = " );
      m_second->render( tw, relativeDepth(depth) );

      if( depth < 0 )
      {
         // it's a statement
         tw->write(" )");
      }
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}

const String& ExprAssign::exprName() const
{
   static String name("=");
   return name;
}

void ExprAssign::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprAssign* self = static_cast<const ExprAssign*>(ps);

   // Generate the values
   CodeFrame& cf = ctx->currentCode();
   TRACE2( "Apply \"%s\" (%d/1)", self->describe().c_ize(), cf.m_seqId );
   
   // Resolve...
   fassert( self->m_first != 0 );
   fassert( self->m_first->lvalueStep() != 0 );
   fassert( self->m_second != 0 );
   
   switch( cf.m_seqId )
   {
      case 0: 
         // check the start.
         cf.m_seqId = 1;
         if( ctx->stepInYield( self->m_second, cf ) )
         {
            return;
         }
         /* no break */
   }
   
   // we're clear -- apply assignment
   ctx->resetCode(self->m_first->lvalueStep());
}

}

/* exprassign.cpp */
