/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_trans.cpp

   Database Interface - SQL Transaction class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 May 2010 00:09:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/dbi_stmt.h>
#include <falcon/dbi_handle.h>

namespace Falcon {


DBIStatement::DBIStatement( DBIHandle *dbh ):
      m_dbh( dbh ),
      m_nLastAffected(-1),
      m_mark(0)
{
}


DBIStatement::~DBIStatement()
{
}

void DBIStatement::gcMark( uint32 mark )
{
   m_mark = mark;
   m_dbh->gcMark(mark);
}


}

/* end of dbi_trans.cpp */
