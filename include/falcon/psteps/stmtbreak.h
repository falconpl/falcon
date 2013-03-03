/*
   FALCON - The Falcon Programming Language.
   FILE: stmtbreak.h

   Statatement -- break (loop break)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:51:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STMTBREAK_H
#define FALCON_STMTBREAK_H

#include <falcon/statement.h>

namespace Falcon
{

/** Break statement.
 *
 * Unrolls to the topmost loop PStep and post a Break item in the data stack.
 */
class FALCON_DYN_CLASS StmtBreak: public Statement
{
public:
   StmtBreak( int32 line=0, int32 chr = 0 );
   StmtBreak( const StmtBreak& other );   
   virtual ~StmtBreak() {};

   virtual void describeTo( String& tgt, int depth=0 ) const;
   virtual void oneLinerTo( String& tgt ) const;
   
   virtual StmtBreak* clone() const{ return new StmtBreak(*this); }
protected:
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of stmtbreak.h */
