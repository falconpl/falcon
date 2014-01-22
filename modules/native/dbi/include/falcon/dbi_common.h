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

#include <falcon/dbi_inbind.h>
#include <falcon/dbi_outbind.h>
#include <falcon/dbi_error.h>
#include <falcon/dbi_handle.h>
#include <falcon/dbi_params.h>
#include <falcon/dbi_recordset.h>
#include <falcon/dbi_stmt.h>
#include <falcon/dbi_refcount.h>

namespace Falcon {

class String;
class VMachine;
class Item;

bool dbi_itemToSqlValue( const Item &item, String &value, char quoteChr='\'' );
void dbi_escapeString( const String& input, String& value, char quoteChr='\'' );
bool dbi_sqlExpand( const String& input, String& output, const ItemArray& arr );

}

#endif

/* end of dbi_common.h */
