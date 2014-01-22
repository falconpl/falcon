/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_classstatement.cpp
 *
 * DBI Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Tue, 21 Jan 2014 16:38:11 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#define SRC "modules/native/dbi/dbi/dbi_classstatement.cpp"

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include "dbi.h"
#include "dbi_mod.h"
#include "dbi_classstatement.h"

namespace Falcon {
namespace DBI {

namespace {

   /*#
      @method execute Statement
      @brief Executes a repeated statement.
      @optparam ... The data to be passed to the repeated statement.
      @return An instance of @a Recorset if the query generated a recorset.
      @raise DBIError if the database engine reports an error.

      This method executes a statement that has been prepared through
      the @a Handle.prepare method. If the prepared statement
      could return a recordset, it is returned.

      The number of affected rows will be stored also in the @a Statement.affected property.

      @see Handle.prepare
   */
   FALCON_DECLARE_FUNCTION( execute, "..." );
   FALCON_DEFINE_FUNCTION_P(execute)
   {
      DBIStatement *dbt = ctx->tself<DBIStatement*>();
      DBIRecordset* res = 0;

      if( pCount != 0 )
      {
         ItemArray params( pCount );
         for( int32 i = 0; i < pCount; i++)
         {
            params.append( *ctx->param(i) );
         }
         res = dbt->execute( &params );
      }
      else {
         res = dbt->execute();
      }

      if( res != 0 )
      {
         DBIModule* mod = static_cast<DBIModule*>(methodOf()->module());
         Class* cls = mod->recordsetClass();
         ctx->returnFrame( FALCON_GC_STORE(cls,res) );
      }
      else
      {
         ctx->returnFrame();
      }
   }

   /*#
      @method aexec Statement
      @brief Executes a repeated statement.
      @optparam params The data to be passed to the repeated statement.
      @return An instance of @a Recorset if the query generated a recorset.
      @raise DBIError if the database engine reports an error.

      This method executes a statement that has been prepared through
      the @a Handle.prepare method. If the prepared statement
      could return a recordset, it is returned.

      Instead of passing the extra parameters to the underlying
      query, this method sends the value of a single array paramter.

      The number of affected rows will be stored also in the @a Statement.affected property.

      @see Handle.prepare
      @see Handle.execute
   */

   FALCON_DECLARE_FUNCTION(aexec, "params:[A]" );
   FALCON_DEFINE_FUNCTION_P1(aexec)
   {
      Item* i_params = ctx->param(0);
      if( i_params == 0 || ! i_params->isArray() )
      {
         throw paramError(__LINE__);
      }

      DBIStatement *dbt = ctx->tself<DBIStatement *>();
      DBIRecordset* res = dbt->execute( i_params->asArray() );

      if( res != 0 )
      {
         DBIModule* mod = static_cast<DBIModule*>(methodOf()->module());
         Class* cls = mod->recordsetClass();
         ctx->returnFrame( FALCON_GC_STORE(cls,res) );
      }
      else
      {
         ctx->returnFrame();
      }
   }


   /*# @property affected Statement

      Indicates the amount of rows affected by the last query performed on this
      statement (through the @a Statement.execute method).

      Will be 0 if none, -1 if unknown, or a positive value if the number of
      rows can be determined.
   */
   void Statement_affected(const Class*, const String&, void *instance, Item &property )
   {
       DBIStatement *dbt = static_cast<DBIStatement *>( instance );
       property = dbt->affectedRows();
   }


   /*#
      @method reset Statement
      @brief Resets this statement
      @raise DBIError if the statement cannot be reset.

      Some Database engines allow to reset a statement and retry to issue (execute) it
      without re-creating it anew.

      If the database engine doesn't support this feature, a DBIError will be thrown.
   */

   FALCON_DECLARE_FUNCTION(reset, "");
   FALCON_DEFINE_FUNCTION_P1(reset)
   {
      DBIStatement *dbt = ctx->tself<DBIStatement *>();
      dbt->reset();
      ctx->returnFrame();
   }

   /*#
       @method close Statement
       @brief Close this statement.

       Statements are automatically closed when the statement object
       is garbage collected, but calling explicitly this helps to
       reclaim data as soon as it is not necessary anymore.
   */
   FALCON_DECLARE_FUNCTION(close, "");
   FALCON_DEFINE_FUNCTION_P1(close)
   {
      DBIStatement *dbt = ctx->tself<DBIStatement *>();
      dbt->close();
      ctx->returnFrame();
   }
}

//========================================================================
// Class satatement
//========================================================================

ClassStatement::ClassStatement():
      Class("%Statement")
{
}

ClassStatement::~ClassStatement()
{}

void ClassStatement::dispose( void* instance ) const
{
   DBIStatement* dbh = static_cast<DBIStatement*>(instance);
   delete dbh;
}

void* ClassStatement::clone( void* ) const
{
   return 0;
}

void* ClassStatement::createInstance() const
{
   return 0;
}

void ClassStatement::gcMarkInstance( void* instance, uint32 mark ) const
{
   DBIStatement* dbs = static_cast<DBIStatement*>(instance);
   dbs->gcMark(mark);
}

bool ClassStatement::gcCheckInstance( void* instance, uint32 mark ) const
{
   DBIStatement* dbs = static_cast<DBIStatement*>(instance);
   return dbs->currentMark() >= mark;
}

}
}

/* end of dbi_classstatement.cpp */
