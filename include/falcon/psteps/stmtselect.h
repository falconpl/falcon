/*
   FALCON - The Falcon Programming Language.
   FILE: select.h

   Syntactic tree item definitions -- select statement.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 15 Aug 2011 13:58:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SELECT_H_
#define _FALCON_SELECT_H_

#include <falcon/psteps/switchlike.h>

namespace Falcon {

class Symbol;
class Expression;
class VMContext;

/** Handler for the select statement.

 The select statements selects a branch depending on the type ID or on the
 class of the result of an expression.

 The try statement uses an expressionless select to account for catch clauses.
 For the same reason, branch declarator have a "target symbol" that is not used
 in the "select" statement, but may be used in catch clauses to store the
 incoming caught value.
 */
class FALCON_DYN_CLASS StmtSelect: public SwitchlikeStatement
{
public:
   /** Create the select statement.
    \param line The line where this statement is declared in the source.
    \param chr The character at which this statement is declared in the source.

    The expression may be left to 0 if this instance is not meant to be added
    to a syntactic tree directly, but it's used just as a dictionary of
    type selectors.

    This is the case of try/catch, but it might be used by third party code
    for similar reasons.
    */
   StmtSelect( int32 line = 0, int32 chr = 0 );

   StmtSelect( const StmtSelect& other );
   virtual ~StmtSelect();

   virtual void renderHeader( TextWriter* tw, int32 depth ) const;
   virtual StmtSelect* clone() const { return new StmtSelect(*this); }

private:
   static void apply_( const PStep*, VMContext* ctx );
};

   
   
}

#endif

/* end of stmtselect.h */
