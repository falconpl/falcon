/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_ext.h
 *
 * DBI Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Sun, 23 Dec 2007 22:02:37 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/setup.h>
#include <falcon/types.h>

#ifndef DBI_EXT_H
#define DBI_EXT_H

namespace Falcon {

class VMachine;

namespace Ext {

//=====================
// DBI Generic
//=====================

void DBIConnect( VMachine *vm );

//=====================
// DBIBaseTrans
//=====================

void Transaction_query( VMachine *vm );
void Transaction_call( VMachine *vm );
void Transaction_prepare( VMachine *vm );
void Transaction_execute( VMachine *vm );
void Transaction_commit( VMachine *vm );
void Transaction_rollback( VMachine *vm );
void Transaction_tropen( VMachine *vm );
void Transaction_close( VMachine *vm );
void Transaction_getLastID( VMachine *vm );

//=====================
// DBI Handle
//=====================

void Handle_tropen( VMachine *vm );
void Handle_close( VMachine *vm );


//=====================
// DBI Recordset
//=====================

void Recordset_discard( VMachine *vm );
void Recordset_fetch( VMachine *vm );
void Recordset_do( VMachine *vm );

void Recordset_getCurrentRow( VMachine *vm );
void Recordset_getRowCount( VMachine *vm );
void Recordset_getColumnCount( VMachine *vm );
void Recordset_getColumnNames( VMachine *vm );
void Recordset_close( VMachine *vm );

//=====================
// DBI Error class
//=====================

void DBIError_init( VMachine *vm );

}
}

#endif

/* end of dbi_ext.h */

