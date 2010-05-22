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

#include <falcon/dbi_trans.h>

namespace Falcon {


DBITransaction::DBITransaction( DBIHandle *dbh, DBISettingParams* params ):
      m_dbh( dbh ),
      m_settings( params )
{
}


DBITransaction::~DBITransaction()
{
}


FalconData *DBITransaction::clone() const
{
   return 0;
}


void DBITransaction::gcMark( uint32 v )
{
}


}

/* end of dbi_trans.cpp */
