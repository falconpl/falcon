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
   {}
   
   virtual ~Statement();

   /** Subclasses can set this to true to be discareded during parsing.*/
   inline bool discardable() const { return m_discardable; }
   
protected:
   bool m_discardable;

   friend class SynTree;
   friend class RuleSynTree;
};

}

#endif

/* end of statement.h */
