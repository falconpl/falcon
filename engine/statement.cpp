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

#include <falcon/trace.h>
#include <falcon/statement.h>
#include <falcon/expression.h>
#include <falcon/syntree.h>
#include <falcon/vm.h>
#include <falcon/stderrors.h>

namespace Falcon
{
Statement::~Statement() 
{}


TreeStep* Statement::minimize_basic(TreeStep* ts)
{
   if( ts == 0 ) {
      return 0;
   }

   if( ts->category() == TreeStep::e_cat_syntree ) {
      SynTree* st = static_cast<SynTree*>(ts);
      int arity = st->arity();

      if( arity == 0 ) {
         delete st;
         return 0;
      }
      else if( arity == 1 ) {
         TreeStep* ret = st->detach(0);
         delete st;
         ret->setParent(this);
         return ret;
      }
   }

   return ts;
}

}

/* end of statement.cpp */
