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
#include <falcon/vm.h>

namespace Falcon
{

StmtWhile::StmtWhile( Expression* check, SynTree* stmts ):
   Statement( while_t ),
   m_check(check),
   m_stmts( stmts )
{}

StmtWhile::~StmtWhile()
{
   delete m_check;
   delete m_stmts;
}

void StmtWhile::toString( String& tgt ) const
{
   tgt = "while " + m_check->toString() + "\n" +
           m_stmts->toString() + "\n" +
         "end\n";
}


void StmtWhile::perform( VMachine* vm ) const
{
   // call me back...
   vm->pushCode( this );
   // after evaluating the expression.
   m_check->perform();
}


void StmtWhile::apply( VMachine* vm ) const
{
   if ( vm->topData()->isTrue() )
   {
      // redo.
      m_check->perform();
      vm->pushCode( m_stmts );
   }
   else {
      //we're done
      vm->popCode();
   }

   // either way, the evaluated value isn't needed.
   vm->popData();
}



StmtIf::StmtIf( Expression* check, SynTree ifTrue, SynTree ifFalse ):
      Statement( if_t ),
      m_check( check ),
      m_ifTrue( ifTrue ),
      m_ifFalse( ifFalse )
{
}


StmtIf::~StmtIf()
{
   delete m_check;
   delete m_ifTrue;
   delete m_ifFalse;
}


void StmtIf::toString( String& tgt ) const
{
   tgt = "if "+ m_check->toString() + "\n" + m_ifTrue->toString();

   if( m_ifFalse != 0  ) {
      tgt += "else\n" + m_ifFalse->toString();
   }
   tgt += "end\n";
}


virtual void StmtIf::perform( VMachine* vm ) const
{
   vm->pushCode( this );
   m_check->perform();
}


virtual void StmtIf::apply( VMachine* vm ) const
{
   if ( vm->topData().isTrue() )
   {
      // we're gone -- but we may use our frame.
      vm->currentCode()->m_pstep = m_ifTrue;
   }
   else {
      //we're gone -- but can we use our frame?
      if( m_ifFalse != 0 )
      {
         vm->currentCode()->m_pstep = m_ifFalse;
      }
      else
      {
         vm->popCode();
      }
   }

  // either way, the evaluated value isn't needed.
  vm->popData();
}

}

/* end of statement.cpp */
