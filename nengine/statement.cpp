/*
   FALCON - The Falcon Programming Language.
   FILE: statement.cpp

   Syntactic tree item definitions -- statements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/statement.h>
#include <falcon/expression.h>
#include <falcon/vm.h>

namespace Falcon
{


StmtAutoexpr::StmtAutoexpr( Expression* expr ):
      Statement(autoexpr_t),
      m_expr( expr )
{
   m_expr->precompile(&m_pcExpr);
}

StmtAutoexpr::~StmtAutoexpr()
{
   delete m_expr;
}

void StmtAutoexpr::toString( String& tgt ) const
{
   m_expr->toString( tgt );
}


void StmtAutoexpr::perform( VMachine* vm ) const
{
   vm->pushCode( this );
   // TODO: use the pre-compiled version.
   // I am using perform here to check perform works as well as precompile
   //m_expr->perform( vm );
   m_pcExpr.perform(vm);
}

void StmtAutoexpr::apply( VMachine* vm ) const
{
   // remove ourself and the data left by the expression.
   vm->popCode();
   vm->popData();
}

//====================================================================
//

StmtWhile::StmtWhile( Expression* check, SynTree* stmts ):
   Statement( while_t ),
   m_check(check),
   m_stmts( stmts )
{
   check->precompile(&m_pcCheck);
}

StmtWhile::~StmtWhile()
{
   delete m_check;
   delete m_stmts;
}

void StmtWhile::toString( String& tgt ) const
{
   tgt = "while " + m_check->toString() + "\n" +
           m_stmts->toString() +
         "end\n";
}


void StmtWhile::perform( VMachine* vm ) const
{
   // call me back...
   vm->pushCode( this );
   // after evaluating the expression.
   m_pcCheck.perform(vm);
}


void StmtWhile::apply( VMachine* vm ) const
{
   if ( vm->topData().isTrue() )
   {
      // redo.
      //m_check->perform(vm);
      m_pcCheck.perform(vm);
      vm->pushCode( m_stmts );
   }
   else {
      //we're done
      vm->popCode();
   }

   // either way, the evaluated value isn't needed.
   vm->popData();
}



//====================================================================
//

StmtIf::StmtIf( Expression* check, SynTree* ifTrue, SynTree* ifFalse ):
   Statement( if_t )
{
   m_ifFalse = ifFalse;
   addElif( check, ifTrue );
}

StmtIf::~StmtIf()
{
   delete m_ifFalse;
   // elif branches will take care of themselves.
}

StmtIf& StmtIf::addElif( Expression *check, SynTree* ifTrue )
{
   m_elifs.push_back( ElifBranch(check, ifTrue) );
   // Why not compiling it in the constructor?
   // -- because STL vector does a copy constructor here, and that would result
   // -- in filling the PCode twice.
   m_elifs.back().compile();

   return *this;
}

StmtIf& StmtIf::setElse( SynTree* ifFalse )
{
   delete m_ifFalse; // in case it was there.
   m_ifFalse = ifFalse;
   return *this;
}

void StmtIf::toString( String& tgt ) const
{
   tgt = "if "+ m_elifs[0].m_check->toString() + "\n"
              + m_elifs[0].m_ifTrue->toString();

   for ( int i = 1; i < m_elifs.size(); ++i )
   {
      tgt += "elif " + m_elifs[i].m_check->toString() + "\n"
                     + m_elifs[i].m_ifTrue->toString();
   }

   if( m_ifFalse != 0  ) {
      tgt += "else\n" + m_ifFalse->toString();
   }

   tgt += "end\n";
}


void StmtIf::perform( VMachine* vm ) const
{
   vm->pushCode( this );

   // we have more than 0 elifs.
   m_elifs[0].m_pcCheck.perform( vm );
}


void StmtIf::apply( VMachine* vm ) const
{
   int sid = vm->currentCode().m_seqId;
   if ( vm->topData().isTrue() )
   {
      // we're gone -- but we may use our frame.
      vm->resetCode( m_elifs[sid].m_ifTrue );
   }
   else
   {
      // try next else-if
      if( ++sid < m_elifs.size() )
      {
         vm->currentCode().m_seqId = sid;
         m_elifs[sid].m_pcCheck.perform( vm );
      }
      else
      {
         // we're out of elifs.
         if( m_ifFalse != 0 )
         {
            vm->resetCode(m_ifFalse);
         }
         else
         {
            // just pop
            vm->popCode();
         }
      }
   }

  // In any way, the evaluated data is to be discarded
  vm->popData();
}


StmtIf::ElifBranch::~ElifBranch()
{
   delete m_check;
   delete m_ifTrue;
}

void StmtIf::ElifBranch::compile()
{
   m_check->precompile( &m_pcCheck );
}

}

/* end of statement.cpp */
