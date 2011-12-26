/*
   FALCON - The Falcon Programming Language.
   FILE: stmtcontinue.h

   Statatement -- contunue (loop restart)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:51:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STMTCONTINUE_H
#define FALCON_STMTCONTINUE_H

#include <falcon/statement.h>

namespace Falcon
{

/** Continue statement.
 *
 * Unrolls to the topmost continue PStep and proceeds from there.
 */
class FALCON_DYN_CLASS StmtContinue: public Statement
{
public:
   StmtContinue( int32 line=0, int32 chr = 0 );
   virtual ~StmtContinue() {};

   void describeTo( String& tgt, int depth=0 ) const;
   void oneLinerTo( String& tgt ) const;
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of stmtcontinue.h */
