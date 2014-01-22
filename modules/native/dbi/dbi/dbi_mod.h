/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_mod.h
 *
 * Helper/inner functions for DBI base.
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Mon, 13 Apr 2009 18:56:48 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef FALCON_DBI_MOD_H
#define FALCON_DBI_MOD_H

#include <falcon/string.h>
#include <falcon/dbi_handle.h>

namespace Falcon {

int dbh_itemToSqlValue( DBIHandle *dbh, const Item *i, String &value );
int dbh_realSqlExpand( VMContext *vm, DBIHandle *dbh, String &sql, int startAt=0 );
void dbh_escapeString( const String& input, String& value );
void dbh_throwError( const char* file, int line, int code, const String& desc );
void dbh_return_recordset( VMContext *vm, DBIRecordset *rec );

}

#endif

/* end of dbi_mod.h */
