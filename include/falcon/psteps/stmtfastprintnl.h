/*
   FALCON - The Falcon Programming Language.
   FILE: stmtfastprintnl.h

   Syntactic tree item definitions -- Fast Print with new line
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 09 Mar 2013 21:16:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STMTFASTPRINTNL_H
#define FALCON_STMTFASTPRINTNL_H

#include <falcon/psteps/stmtfastprint.h>

namespace Falcon
{

/** Fastprint with newline statement.
 The fastprint statement is a line beginning with ">" or ">>", printing 
 everything that's on the line.
 
 This subclass describes the specifics of the fastprint with
 one new line (nl) ">".

*/
class FALCON_DYN_CLASS StmtFastPrintNL: public StmtFastPrint
{
public:
   StmtFastPrintNL( int line = 0, int chr = 0 );
   StmtFastPrintNL( const StmtFastPrintNL& other );
   virtual ~StmtFastPrintNL();
   virtual StmtFastPrint* clone() const { return new StmtFastPrintNL(*this); }
};

}

#endif

/* end of stmtfastprintnl.h */
