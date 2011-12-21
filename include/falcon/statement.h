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

#include <falcon/pstep.h>
#include <falcon/vmcontext.h>

namespace Falcon
{

class Expression;
class SynTree;

/** Statement.
 * Statements are PStep that may require other sub-sequences to be evaluated.
 * In other words, they are
 */
class FALCON_DYN_CLASS Statement: public PStep
{

public:
   typedef enum {
      e_stmt_breakpoint,
      e_stmt_autoexpr,
      e_stmt_if,
      e_stmt_while,
      e_stmt_return,
      e_stmt_rule,
      e_stmt_cut,
      e_stmt_init,
      e_stmt_for_in,
      e_stmt_for_to,
      e_stmt_continue,
      e_stmt_break,
      e_stmt_fastprint,
      e_stmt_for,
      e_stmt_try,
      e_stmt_raise,
      e_stmt_select,
      custom_t
   } t_statement;

   Statement( t_statement type, int32 line=0, int32 chr=0 ):
      PStep( line, chr ),
      m_discardable(false),
      m_type(type)
   {}

   inline virtual ~Statement() {}
   inline t_statement type() const { return m_type; }
   /** Subclasses can set this to true to be discareded during parsing.*/
   inline bool discardable() const { return m_discardable; }

protected:
   bool m_discardable;

   friend class SynTree;
   friend class RuleSynTree;
private:
   t_statement m_type;
};

}

#endif

/* end of statement.h */
