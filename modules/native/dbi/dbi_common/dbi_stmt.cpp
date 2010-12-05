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

namespace Falcon {


DBIStatement::DBIStatement( DBIHandle *dbh ):
      m_dbh( dbh ),
      m_nLastAffected(-1)
{
}


DBIStatement::~DBIStatement()
{
}


FalconData *DBIStatement::clone() const
{
   return 0;
}


void DBIStatement::gcMark( uint32 v )
{
}


int64 DBIStatement::affectedRows()
{
   return m_nLastAffected;
}

}

/* end of dbi_trans.cpp */
