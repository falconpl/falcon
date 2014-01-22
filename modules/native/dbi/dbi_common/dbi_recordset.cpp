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
#include <falcon/dbi_handle.h>

namespace Falcon {

DBIRecordset::DBIRecordset( DBIHandle* generator ):
      m_dbh( generator ),
      m_mark(0)
{}

DBIRecordset::~DBIRecordset()
{}

DBIRecordset *DBIRecordset::getNext()
{
   return 0;
}

void DBIRecordset::gcMark( uint32 mark )
{
   m_mark = mark;
   m_dbh->gcMark(mark);
}

}

/* end of dbi_recorset.cpp */
