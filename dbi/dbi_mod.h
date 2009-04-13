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

#include <dbiservice.h>


namespace Falcon {

dbi_type *recordset_getTypes( DBIRecordset *recSet );

DBIRecordset *dbh_query_base( DBIBaseTrans* dbh, const String &sql );
int dbh_itemToSqlValue( DBIBaseTrans *dbh, const Item *i, String &value );
int dbh_realSqlExpand( VMachine *vm, DBIBaseTrans *dbh, String &sql, int startAt=0 );
DBIRecordset *dbh_baseQueryOne( VMachine *vm, int startAt = 0 );
void dbh_return_recordset( VMachine *vm, DBIRecordset *rec );


int dbr_getItem( VMachine *vm, DBIRecordset *dbr, dbi_type typ, int cIdx, Item &item );
int dbr_checkValidColumn( VMachine *vm, DBIRecordset *dbr, int cIdx );
void dbr_execute( VMachine *vm, DBIHandle *dbh, const String &sql );
int dbr_getPersistPropertyNames( VMachine *vm, CoreObject *self, String columnNames[], int maxColumnCount );

}

#endif

/* end of dbi_mod.h */
