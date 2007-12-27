/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_ext.h

   DBI Falcon extension interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 Dec 2007 22:02:37 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/setup.h>
#include <falcon/types.h>

#ifndef DBI_EXT_H
#define DBI_EXT_H

namespace Falcon {

class VMachine;

namespace Ext {

FALCON_FUNC DBIConnect( VMachine *vm );

FALCON_FUNC DBIHandle_startTransaction( VMachine *vm );
FALCON_FUNC DBIHandle_close( VMachine *vm );
FALCON_FUNC DBIHandle_query( VMachine *vm );
FALCON_FUNC DBIHandle_execute( VMachine *vm );

FALCON_FUNC DBITransaction_query( VMachine *vm );
FALCON_FUNC DBITransaction_execute( VMachine *vm );
FALCON_FUNC DBITransaction_close( VMachine *vm );

FALCON_FUNC DBIRecordset_next( VMachine *vm );
FALCON_FUNC DBIRecordset_fetchArray( VMachine *vm );
FALCON_FUNC DBIRecordset_fetchDict( VMachine *vm );
FALCON_FUNC DBIRecordset_getRowCount( VMachine *vm );
FALCON_FUNC DBIRecordset_getColumnCount( VMachine *vm );
FALCON_FUNC DBIRecordset_getColumnNames( VMachine *vm );
FALCON_FUNC DBIRecordset_getColumnTypes( VMachine *vm );
FALCON_FUNC DBIRecordset_asString( VMachine *vm );
FALCON_FUNC DBIRecordset_asInteger( VMachine *vm );
FALCON_FUNC DBIRecordset_asInteger64( VMachine *vm );
FALCON_FUNC DBIRecordset_asNumeric( VMachine *vm );
FALCON_FUNC DBIRecordset_asDate( VMachine *vm );
FALCON_FUNC DBIRecordset_asTime( VMachine *vm );
FALCON_FUNC DBIRecordset_asDateTime( VMachine *vm );
FALCON_FUNC DBIRecordset_getLastError( VMachine *vm );
FALCON_FUNC DBIRecordset_close( VMachine *vm );

}
}

#endif

/* end of dbi_ext.h */

