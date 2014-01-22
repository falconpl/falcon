/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_handle.cpp

   Database Interface - Main handle driver
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 May 2010 00:09:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/dbi_handle.h>
#include <falcon/dbi_error.h>
#include <falcon/dbi_common.h>
#include <falcon/itemarray.h>
#include <falcon/itemarray.h>

namespace Falcon {

DBIHandle::DBIHandle( const Class* h):
   m_nLastAffected(-1),
   m_handler(h)
{
}


DBIHandle::~DBIHandle()
{
}

void DBIHandle::sqlExpand( const String& sql, String& tgt, const ItemArray& params )
{
   if( !dbi_sqlExpand( sql, tgt, params ) )
   {
      String temp = "";
      temp.A("Array of ").N((int32)params.length()).A(" -> ");
      temp += sql;

      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_BIND_INTERNAL, __LINE__ )
            .extra( temp ));
   }
}


int64 DBIHandle::affectedRows()
{
   return m_nLastAffected;
}

}

/* end of dbi_handle.h */
