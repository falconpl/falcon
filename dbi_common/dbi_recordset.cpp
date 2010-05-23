/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_recordset.cpp

   Database Interface - SQL Recordset class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 May 2010 00:09:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/dbi_recordset.h>

namespace Falcon {

DBIRecordset::DBIRecordset( DBIHandle* generator ):
      m_dbh( generator )
{}

DBIRecordset::~DBIRecordset()
{}

FalconData *DBIRecordset::clone() const
{
   return 0;
}


void DBIRecordset::gcMark( uint32 v )
{
}

}

/* end of dbi_recorset.cpp */
