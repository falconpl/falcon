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
#include <falcon/item.h>
#include <falcon/carray.h>

namespace Falcon
{

DBIHandle::DBIHandle():
   m_nLastAffected(-1)
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


void DBIHandle::std_result( DBIRecordset* rs, Item& res )
{
	res.setNil();
	try {
		  if( rs != 0 )
		  {
			   if( rs->fetchRow() )
			   {
				   int count = rs->getColumnCount();
				   if( count == 1 ) {
					   rs->getColumnValue(0,res);
				   }
				   else {
					   CoreArray* arr = new CoreArray();
					   arr->resize(count);
					   for( int i = 0; i < count; ++i ) {
						   rs->getColumnValue(i, arr->at(i));
					   }
					   res = arr;
				   }
			   }
		  }

		  delete rs;
	}
	catch( ... ) {
		   delete rs;
		   throw;
	}
}

void DBIHandle::gcMark( uint32 )
{

}

FalconData* DBIHandle::clone() const
{
   return 0;
}

int64 DBIHandle::affectedRows()
{
   return m_nLastAffected;
}

}

/* end of dbi_handle.h */
