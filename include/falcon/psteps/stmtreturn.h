/*
   FALCON - The Falcon Programming Language.
   FILE: stmtreturn.h

   Statatement -- return
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 22:25:08 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_RETURN_H
#define FALCON_RETURN_H

#include <falcon/statement.h>

namespace Falcon
{


/** Return statement.
 *
 * Exits the current function.
 */
class FALCON_DYN_CLASS StmtReturn: public Statement
{
public:
   /** Returns a value */
   StmtReturn( int32 line=0, int32 chr = 0 );
   StmtReturn( Expression* expr, int32 line=0, int32 chr = 0 );
   StmtReturn( const StmtReturn& other );
   virtual ~StmtReturn();

   virtual void describeTo( String& tgt, int depth=0 ) const;
   virtual void oneLinerTo( String& tgt ) const;
   virtual StmtReturn* clone() const { return new StmtReturn(*this); }
   
   bool hasDoubt() const { return m_bHasDoubt; }
   void hasDoubt( bool b );
   
   virtual Expression* selector() const; 
   virtual bool selector( Expression* e );
   
   bool hasEval() const { return m_bHasEval; }
   void hasEval( bool  b ) ;
   
private:
   Expression* m_expr;
   bool m_bHasDoubt;
   bool m_bHasEval;
   
   static void apply_( const PStep*, VMContext* ctx );
   static void apply_expr_( const PStep*, VMContext* ctx );
   static void apply_doubt_( const PStep*, VMContext* ctx );
   static void apply_expr_doubt_( const PStep*, VMContext* ctx );

};

}

#endif

/* end of stmtreturn.h */
