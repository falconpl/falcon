/*
   FALCON - The Falcon Programming Language.
   FILE: stmtraise.h

   Syntactic tree item definitions -- raise.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 15 Aug 2011 23:03:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_STMTRAISE_H_
#define _FALCON_STMTRAISE_H_

#include <falcon/statement.h>

namespace Falcon {

class Expression;

/** Implementation of the raise statement.
 
 \TODO More docs.
 */
class FALCON_DYN_CLASS StmtRaise: public Statement
{
public:
   StmtRaise( int32 line=0, int32 chr = 0 );
   
   /** Raises an expression.
    \param risen The expression generating the sitem to be risen (mandatory).
    \param line The line where this statement is declared.
    \param chr The character where this statement is delcared.
    */
   StmtRaise( Expression* risen, int32 line=0, int32 chr = 0 ); 
   StmtRaise( const StmtRaise& other ); 
   
   virtual ~StmtRaise();

   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual StmtRaise* clone() const { return new StmtRaise(*this); }

   /** Gets the expression generating the item to be raised. 
    */
   TreeStep* expr() const { return m_expr; }

   virtual TreeStep* selector() const;
   virtual bool selector( TreeStep* e );
   
private:
   TreeStep* m_expr;
   
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of stmtraise.h */
