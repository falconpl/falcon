/*
   FALCON - The Falcon Programming Language.
   FILE: stmtfastprintnl.cpp

   Syntactic tree item definitions -- Fast Print statement.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 09 Mar 2013 21:16:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/stmtfastprintnl.cpp"


#include <falcon/expression.h>
#include <falcon/psteps/stmtfastprintnl.h>
#include <falcon/synclasses.h>

namespace Falcon
{
   
StmtFastPrintNL::StmtFastPrintNL( int line, int chr ):
   StmtFastPrint( line, chr, true )
{
   FALCON_DECLARE_SYN_CLASS(stmt_fastprintnl)
   m_bAddNL = true;
   apply = apply_;
}


StmtFastPrintNL::StmtFastPrintNL( const StmtFastPrintNL& other ):
    StmtFastPrint( other )
{
   m_bAddNL = true;
   apply = apply_;
}

StmtFastPrintNL::~StmtFastPrintNL()
{
}

}

/* end of stmtfastprintnl.cpp */
