/*
 * FALCON - The Falcon Programming Language.
 * FILE: sqlite3_ext.cpp
 *
 * SQLite3 Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 16:47:15 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/engine.h>
#include <sqlite3.h>
#include <falcon/function.h>
#include <falcon/vmcontext.h>

#include "sqlite3_mod.h"
#include "sqlite3_ext.h"

/*#
   @beginmodule dbi.sqlite3
*/
namespace Falcon
{
namespace Ext
{

namespace {
/*#
   @class SQLite3
   @brief Direct interface to SQLite3 database.
   @param connect String containing connection parameters.
   @optparam options Default statement options for this connection.
*/

FALCON_DECLARE_FUNCTION(init, "connect:S,options:[S]")
FALCON_DEFINE_FUNCTION_P1(init)
{
   Item *paramsI = ctx->param(0);
   Item *i_tropts = ctx->param(1);
   if (  paramsI == 0 || ! paramsI->isString()
         || ( i_tropts != 0 && ! i_tropts->isString() ) )
   {
      throw paramError(__LINE__ );
   }

   String *params = paramsI->asString();
   DBIHandleSQLite3 *hand = ctx->tself<DBIHandleSQLite3*>();
   try
   {
      hand->connect(*params);
      if( i_tropts != 0 )
      {
         hand->options( *i_tropts->asString() );
      }

      // great, we have the database handler open. Now we must create a falcon object to store it.
      ctx->returnFrame(ctx->self());
   }
   catch( DBIError* error )
   {
      delete hand;
      throw error;
   }
}
}


ClassSqlite3DBIHandle::ClassSqlite3DBIHandle():
         Class("Sqlite3")
{
   setConstuctor( new FALCON_FUNCTION_NAME(init) );
}


ClassSqlite3DBIHandle::~ClassSqlite3DBIHandle()
{
}

void ClassSqlite3DBIHandle::dispose( void* instance ) const
{
   DBIHandleSQLite3* item = static_cast<DBIHandleSQLite3*>(instance);
   delete item;
}

void* ClassSqlite3DBIHandle::clone( void* ) const
{
   return 0;
}


void* ClassSqlite3DBIHandle::createInstance() const
{
   DBIHandleSQLite3* h = new DBIHandleSQLite3(this);
   return h;
}


void ClassSqlite3DBIHandle::gcMarkInstance( void* instance, uint32 mark ) const
{
   DBIHandleSQLite3* dbh = static_cast<DBIHandleSQLite3*>(instance);
   dbh->gcMark(mark);
}


bool ClassSqlite3DBIHandle::gcCheckInstance( void* instance, uint32 mark ) const
{
   DBIHandleSQLite3* dbh = static_cast<DBIHandleSQLite3*>(instance);
   return dbh->currentMark() >= mark;
}

} /* namespace Ext */
} /* namespace Falcon */

/* end of sqlite3_ext.cpp */
