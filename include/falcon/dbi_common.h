/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_common.h
  
   Database Interface - Helper/inner functions for DBI base.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 13 Apr 2009 18:56:48 +0200
  
   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)
  
   See LICENSE file for licensing details.
*/

#ifndef FALCON_DBI_COMMON_H_
#define FALCON_DBI_COMMON_H_

#include <falcon/string.h>

#include <falcon/dbi_bind.h>
#include <falcon/dbi_outbind.h>
#include <falcon/dbi_error.h>
#include <falcon/dbi_handle.h>
#include <falcon/dbi_params.h>
#include <falcon/dbi_recordset.h>
#include <falcon/dbi_trans.h>

namespace Falcon {

class String;
class VMachine;
class Item;

int dbi_itemToSqlValue( const Item &item, String &value );
void dbi_escapeString( const String& input, String& value );
void dbi_throwError( const char* file, int line, int code, const String& desc );
void dbi_return_recordset( VMachine *vm, DBIRecordset *rec );

}

#endif

/* end of dbi_common.h */
