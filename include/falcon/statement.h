/*
   FALCON - The Falcon Programming Language.
   FILE: statement.h

   Syntactic tree item definitions -- statements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STATEMENT_H
#define FALCON_STATEMENT_H

#include <falcon/treestep.h>
#include <falcon/vmcontext.h>

namespace Falcon
{

class Expression;
class SynTree;

/** Statement.
 * Statements are PStep that may require other sub-sequences to be evaluated.
 * In other words, they are
 */
class FALCON_DYN_CLASS Statement: public TreeStep
{
public:
   
   Statement( int32 line=0, int32 chr=0 ):
      TreeStep( TreeStep::e_cat_statement, line, chr ),
      m_discardable(false)
   {
      // all the statements are composed, as they parse their expressions separately
      m_bIsComposed = true;
   }
   
   virtual ~Statement();

   /**
    * Called when the statement is complete after a source compilation.
    *
    * Statements are usually created with some default data (i.e. a child
    * syntree block) that can be removed. This method is meant to clean
    * the defaults that a compiler may leave behind.
    *
    * By default does nothing.
    */
   virtual void minimize() {}

   /** Subclasses can set this to true to be discareded during parsing.*/
   inline bool discardable() const { return m_discardable; }
   
protected:
   bool m_discardable;

   TreeStep* minimize_basic( TreeStep* source );

   friend class SynTree;
   friend class RuleSynTree;
};

}

#endif

/* end of statement.h */
