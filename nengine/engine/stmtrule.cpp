/*
   FALCON - The Falcon Programming Language.
   FILE: stmtrule.cpp

   Syntactic tree item definitions -- statements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 02 Apr 2011 13:52:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/stmtrule.h>
#include <falcon/rulesyntree.h>
#include <falcon/vm.h>
#include <falcon/trace.h>

namespace Falcon
{

StmtRule::StmtRule( int32 line, int32 chr )
{
   apply = apply_;

   // create a base rule syntree
   m_altTrees.push_back( RuleSynTree() );

   // push ourselves when prepare is invoked
   m_step0 = this;
}


StmtRule::~StmtRule()
{
}


StmtRule& StmtRule::addStatement( Statement* stmt )
{
   m_altTrees.back().append( stmt );
   return *this;
}


StmtRule& StmtRule::addAlternative()
{
   m_altTrees.push_back( RuleSynTree() );
   return *this;
}


void StmtRule::describe( String& tgt ) const
{
   tgt += "rule\n";
   bool bFirst = true;
   AltTrees::iterator iter = m_altTrees.begin();
   while( iter != m_altTrees.end() )
   {
      if( ! bFirst )
      {
         tgt += "or\n";
      }
      bFirst = false;
      iter->describe(tgt);
      ++iter;
   }
   tgt += "end";
}


void StmtRule::apply_( const PStep*s1 , VMachine* vm )
{
   register VMContext* ctx = vm->currentContext();
   const StmtRule* self = static_cast<const StmtRule*>(s1);
   CodeFrame& cf = ctx->currentCode();

   // Always process the first alternative
   if ( cf.m_seqId > 0 && ( ! ctx->regA().isBoolean() || ctx->regA().asBoolean() == true) )
   {
      TRACE1( "Apply 'rule' at line %d -- success ", self->line() );
      // force A to be true
      ctx->regA().setBoolean(true);
      //we're done
      ctx->popCode();
   }
   else
   {
      // on first alternative -- or if previous alternative failed...

      if( cf.m_seqId >= self->m_altTrees.size() )
      {
         // we failed, and we have no more alternatives.
         TRACE1( "Apply 'rule' at line %d -- rule failed", self->line() );

         // we're done
         ctx->popCode();

         // force A to be false
         ctx->regA().setBoolean(false);
      }
      else
      {
         // we have some more alternative to try
         ctx->pushCode( self->m_altTrees[cf.m_seqId++] );
         TRACE1( "Apply 'rule' at line %d -- applying next branch %d",
               self->line(), cf.m_seqId );
      }
   }
}

}

/* end of stmtrule.cpp */
