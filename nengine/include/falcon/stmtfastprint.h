/*
   FALCON - The Falcon Programming Language.
   FILE: stmtfastprint.h

   Syntactic tree item definitions -- Fast Print statement.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 14 Aug 2011 12:41:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STMTFASTPRINT_H
#define FALCON_STMTFASTPRINT_H

#include <falcon/statement.h>

namespace Falcon
{

/** Fastprint statement.
 The fastprint statement is a line beginning with ">" or ">>", printing 
 everything that's on the line.
 
*/
class FALCON_DYN_CLASS StmtFastPrint: public Statement
{
public:
   StmtFastPrint( bool bAddNL = true );
   virtual ~StmtFastPrint();
   
   void add( Expression* expr );
   Expression* at( int n ) const;
   length_t size() const;
   
protected:
   class Private;
   Private* _p;
   
   bool m_bAddNL;
   
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of stmtfastprint.h */
