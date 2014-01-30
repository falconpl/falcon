/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_driverclass.cpp
 *
 * SQLite3 Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Thu, 30 Jan 2014 13:47:51 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#define SRC "modules/native/dbi/dbi_common/dbi_driverclass.cpp"

#include <falcon/engine.h>
#include <falcon/vmcontext.h>
#include <falcon/function.h>

#include <falcon/dbi_handle.h>
#include <falcon/dbi_driverclass.h>
#include <falcon/dbi_error.h>

/*#
   @beginmodule dbi.sqlite3
*/
namespace Falcon
{

namespace {

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
   DBIHandle* hand = ctx->tself<DBIHandle*>();

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


ClassDriverDBIHandle::ClassDriverDBIHandle( const String& name ):
         Class(name)
{
   setConstuctor( new FALCON_FUNCTION_NAME(init) );
}


ClassDriverDBIHandle::~ClassDriverDBIHandle()
{
}

void ClassDriverDBIHandle::dispose( void* instance ) const
{
   DBIHandle* item = static_cast<DBIHandle*>(instance);
   delete item;
}

void* ClassDriverDBIHandle::clone( void* ) const
{
   return 0;
}


void ClassDriverDBIHandle::gcMarkInstance( void* instance, uint32 mark ) const
{
   DBIHandle* dbh = static_cast<DBIHandle*>(instance);
   dbh->gcMark(mark);
}


bool ClassDriverDBIHandle::gcCheckInstance( void* instance, uint32 mark ) const
{
   DBIHandle* dbh = static_cast<DBIHandle*>(instance);
   return dbh->currentMark() >= mark;
}

} /* namespace Falcon */

/* end of dbi_driverclass.cpp */
