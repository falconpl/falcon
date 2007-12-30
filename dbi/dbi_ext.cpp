/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_ext.cpp
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
 * In order to use this file in its compiled form, this source or
 * part of it you have to read, understand and accept the conditions
 * that are stated in the LICENSE file that comes boundled with this
 * package.
 */

#include <stdio.h>
#include <string.h>

#include <falcon/engine.h>
#include <falcon/error.h>

#include "dbi.h"
#include "dbi_ext.h"
#include "../include/dbiservice.h"

namespace Falcon {
namespace Ext {

FALCON_FUNC DBIConnect( VMachine *vm )
{
   Item *paramsI = vm->param(0);
   if (  paramsI == 0 || ! paramsI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   String *params = paramsI->asString();
   String provName = *params;
   String connString = "";
   uint32 colonPos = params->find( ":" );

   if ( colonPos != csh::npos ) {
      provName = params->subString( 0, colonPos );
      connString = params->subString( colonPos + 1 );
   }

   DBIService *provider = theDBIService.loadDbProvider( vm, provName );
   if ( provider != 0 ) {
      // if it's 0, the service has already raised an error in the vm and we have nothing to do.
      String connectErrorMessage;
      dbi_status status;
      DBIHandle *hand = provider->connect( connString, false, status, connectErrorMessage );
      if ( hand == 0 ) {
         if ( connectErrorMessage.length() == 0 )
            connectErrorMessage = "An unknown error has occured during connect";

         vm->raiseModError( new DBIError( ErrorParam( status, __LINE__ )
                                          .desc( connectErrorMessage ) ) );

         return;
      }

      // great, we have the database handler open. Now we must create a falcon object to store it.
      CoreObject *instance = provider->makeInstance( vm, hand );
      vm->retval( instance );
   }

   // no matter what we return if we had an error.
}

/**********************************************************
   Handler class
 **********************************************************/
FALCON_FUNC DBIHandle_startTransaction( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   DBITransaction *trans = dbh->startTransaction();
   if ( trans == 0 ) {
      // raise an error depending on dbh->getLastError();
      return;
   }

   Item *trclass = vm->findGlobalItem( "%DBITransaction" );
   fassert( trclass != 0 && trclass->isClass() );

   CoreObject *oth = trclass->asClass()->createInstance();
   oth->setUserData( trans );
   vm->retval( oth );
}

FALCON_FUNC DBIHandle_query( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   Item *sqlI = vm->param( 0 );
   if ( sqlI == 0 || ! sqlI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   dbi_status retval;
   DBIRecordset *recSet = dbh->query( *sqlI->asString(), retval );

   if ( retval != dbi_ok ) {
      String errorMessage;
      dbh->getLastError( errorMessage );

      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                       .desc( errorMessage ) ) );
      return;
   }

   Item *rsclass = vm->findGlobalItem( "%DBIRecordset" );
   fassert( rsclass != 0 && rsclass->isClass() );

   CoreObject *oth = rsclass->asClass()->createInstance();
   oth->setUserData( recSet );
   vm->retval( oth );
}

FALCON_FUNC DBIHandle_execute( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   Item *sqlI = vm->param( 0 );
   if ( sqlI == 0 || ! sqlI->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   dbi_status retval;
   int affectedRows = dbh->execute( *sqlI->asString(), retval );

   if ( retval != dbi_ok ) {
      String errorMessage;
      dbh->getLastError( errorMessage );
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                       .desc( errorMessage ) ) );
   }

   vm->retval( affectedRows );
}

FALCON_FUNC DBIHandle_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   dbh->close();
}

FALCON_FUNC DBIHandle_getLastError( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   String value;
   dbi_status retval = dbh->getLastError( value );
   if ( retval != dbi_ok ) {
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                      .desc( "Could not get last error message " ) ) );
      return;
   }

   GarbageString *gs = new GarbageString( vm );
   gs->bufferize( value );

   vm->retval( gs );
}

FALCON_FUNC DBIHandle_sqlExpand( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   switch ( dbh->getQueryExpansionCapability() ) {
   case DBIHandle::q_dollar_sign_expansion:
      // TODO: Build array and ship off to query method
      return;

   case DBIHandle::q_question_mark_expansion:
      // TODO: Convert $1, $2 into ?, ? and ship off to query method
      return;

   default:
      // We will handle that below
      break;
   }

   Item *sqlI = vm->param( 0 );

   if ( sqlI == 0 || ! sqlI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   GarbageString *sql = new GarbageString( vm, *sqlI->asString() );
   uint32 dollarPos = sql->find( "$", 0 );

   while ( dollarPos != csh::npos ) {
      int64 pIdx = -1;

      if ( dollarPos == sql->length() - 1 ) {
         vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_error,
                                                     __LINE__ )
                                         .desc( "Stray $ charater at the end of query" ) ) );
         return;
      } else {
         if ( sql->getCharAt( dollarPos + 1 ) == '$' ) {
            sql->remove( dollarPos, 1 );
            dollarPos = sql->find( "$", dollarPos + 1 );
            continue;
         }

         AutoCString asTmp( sql->subString( dollarPos + 1 ) );
         pIdx = atoi( asTmp.c_str() );

         if ( pIdx == 0 ) {
            String s( sql->subString( dollarPos ) );
            s.prepend( "Failed to parse dollar expansion starting at: " );

            vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_error,
                                                        __LINE__ )
                                            .desc( s ) ) );
            return;
         }
      }

      int dollarSize = 2;
      if ( pIdx > 9 ) dollarSize++;

      Item *i = vm->param( pIdx );
      if ( i == 0 ) {
         char errorMessage[128];
         snprintf( errorMessage, 128, "Positional argument (%i) is out of range", pIdx );

         GarbageString *s = new GarbageString( vm );
         s->bufferize( errorMessage );
         vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_error,
                                                     __LINE__ )
                                         .desc( *s ) ) );
         return;
      }

      String value;

      // TODO: handle timestamp values
      // TODO: handle other values via toString method
      if ( i->isInteger() )
         value.writeNumber( i->asInteger() );
      else if ( i->isNumeric() )
         value.writeNumber( i->asNumeric(), "%f" );
      else if ( i->isString() ) {
         dbh->escapeString( *i->asString(), value );
         value.prepend( "'" );
         value.append( "'" );
      }

      sql->insert( dollarPos, 2, value );

      dollarPos = sql->find( "$", dollarPos + dollarSize );
   }

   vm->retval( sql );
}

/**********************************************************
 * Transaction class
 **********************************************************/

FALCON_FUNC DBITransaction_query( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   Item *i_query = vm->param(0);
   if( i_query == 0 || ! i_query->isString() ) {
      // raise error
      return;
   }

   /*
    if ( dbt->query( *i_query->asString() ) != dbi_ok )
    {
        // raise error
    }
    */

   vm->retval(0); // or anything you want to return
}

FALCON_FUNC DBITransaction_execute( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   vm->retval( 0 );
}

FALCON_FUNC DBITransaction_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   vm->retval( 0 );
}

/******************************************************************************
 * Recordset class
 *****************************************************************************/

FALCON_FUNC DBIRecordset_next( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   vm->retval( dbr->next() );
}

int DBIRecordset_getItem( VMachine *vm, DBIRecordset *dbr, dbi_type typ, int cIdx, Item &item )
{
   switch ( typ )
   {
   case dbit_string:
      {
         String value;
         dbi_status retval = dbr->asString( cIdx, value );
         switch ( retval )
         {
         case dbi_ok:
            {
               GarbageString *gsValue = new GarbageString( vm );
               gsValue->bufferize( value );

               item.setString( gsValue );
            }
            break;

         case dbi_nil_value:
            break;

         default:
            // TODO: handle error
            return 0;
         }
      }
      break;

   case dbit_integer:
      {
         int32 value;
         if ( dbr->asInteger( cIdx, value ) != dbi_nil_value )
            item.setInteger( (int64) value );
      }
      break;
   
   case dbit_integer64:
      {
         int64 value;
         if ( dbr->asInteger64( cIdx, value ) != dbi_nil_value )
            item.setInteger( value );
      }
      break;
   
   case dbit_numeric:
      {
         numeric value;
         if ( dbr->asNumeric( cIdx, value ) != dbi_nil_value )
            item.setNumeric( value );
      }
      break;
   
   case dbit_date:
      {
         TimeStamp *ts = new TimeStamp();
         if ( dbr->asDate( cIdx, *ts ) != dbi_nil_value ) {
            Item *ts_class = vm->findGlobalItem( "TimeStamp" );
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            value->setUserData( ts );
            item.setObject( value );
         }
      }
      break;
   
   case dbit_time:
      {
         TimeStamp *ts = new TimeStamp();
         if ( dbr->asTime( cIdx, *ts ) != dbi_nil_value ) {
            Item *ts_class = vm->findGlobalItem( "TimeStamp" );
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            value->setUserData( ts );
            item.setObject( value );
         }
      }
      break;
   
   case dbit_datetime:
      {
         TimeStamp *ts = new TimeStamp();
         if ( dbr->asDateTime( cIdx, *ts ) != dbi_nil_value ) {
            Item *ts_class = vm->findGlobalItem( "TimeStamp" );
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            value->setUserData( ts );
            item.setObject( value );
         }
      }
      break;

   default:
      return 0;
   }

   return 1;
}

FALCON_FUNC DBIRecordset_fetchArray( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   dbi_status nextRetVal = dbr->next();
   switch ( nextRetVal )
   {
   case dbi_ok:
      break;

   case dbi_eof:
      vm->retnil();
      return ;

   default:
      {
         String errorMessage;
         dbr->getLastError( errorMessage );

         vm->raiseModError( new DBIError( ErrorParam( nextRetVal, __LINE__ )
                                         .desc( errorMessage ) ) );
         return ;
      }
   }

   int cCount = dbr->getColumnCount();
   dbi_type cTypes[cCount];
   CoreArray *ary = new CoreArray( vm, cCount );
   
   dbr->getColumnTypes( cTypes );

   for ( int cIdx = 0; cIdx < cCount; cIdx++ )
   {
      Item i;
      if ( DBIRecordset_getItem( vm, dbr, cTypes[cIdx], cIdx, i ) ) {
         ary->append( i );
      } else {
         // TODO: handle error
      }
   }

   vm->retval( ary );
}

FALCON_FUNC DBIRecordset_fetchDict( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   dbi_status nextRetVal = dbr->next();
   switch ( nextRetVal )
   {
   case dbi_ok:
      break;

   case dbi_eof:
      vm->retnil();
      return ;

   default:
      // TODO: Handle error
      break;
   }

   int cCount = dbr->getColumnCount();
   CoreDict *dict = new PageDict( vm, cCount );
   char **cNames = (char **) malloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );
   dbi_type cTypes[cCount];
   
   dbr->getColumnTypes( cTypes );
   dbr->getColumnNames( cNames );

   for ( int cIdx = 0; cIdx < cCount; cIdx++ )
   {
      dbi_status retval;
      GarbageString *gsName = new GarbageString( vm );
      gsName->bufferize( cNames[cIdx] );

      Item i;
      if ( DBIRecordset_getItem( vm, dbr, cTypes[cIdx], cIdx, i ) )
      {
         dict->insert( gsName, i );
      } else {
         // TODO: handle error
      }

      free( cNames[cIdx] );
   }

   free( cNames );

   vm->retval( dict );
}

FALCON_FUNC DBIRecordset_fetchObject( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *oI = vm->param( 0 );
   if ( oI == 0 || ! oI->isObject() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   CoreObject *o = oI->asObject();

   dbi_status nextRetVal = dbr->next();
   switch ( nextRetVal )
   {
   case dbi_ok:
      break;

   case dbi_eof:
      vm->retnil();
      return ;

   default:
      // TODO: Handle error
      break;
   }

   int cCount = dbr->getColumnCount();
   CoreDict *dict = new PageDict( vm, cCount );
   char **cNames = (char **) malloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );
   dbi_type cTypes[cCount];

   dbr->getColumnTypes( cTypes );
   dbr->getColumnNames( cNames );

   for ( int cIdx = 0; cIdx < cCount; cIdx++ ) {
      Item i;
      if ( DBIRecordset_getItem( vm, dbr, cTypes[cIdx], cIdx, i ) )
      {
         o->setProperty( cNames[cIdx], i );
      } else {
         // TODO: handle error
      }

      free( cNames[cIdx] );
   }

   free( cNames );

   vm->retval( o );
}


FALCON_FUNC DBIRecordset_getRowCount( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   vm->retval( dbr->getRowCount() );
}

FALCON_FUNC DBIRecordset_getColumnTypes( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   int cCount = dbr->getColumnCount();
   CoreArray *ary = new CoreArray( vm, cCount );
   dbi_type cTypes[cCount];

   dbr->getColumnTypes( cTypes );

   for (int cIdx=0; cIdx < cCount; cIdx++ )
      ary->append( (int64) cTypes[cIdx] );

   vm->retval( ary );
}

FALCON_FUNC DBIRecordset_getColumnNames( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   int cCount = dbr->getColumnCount();
   CoreArray *ary = new CoreArray( vm, cCount );
   char **cNames = (char **) malloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );

   dbr->getColumnNames( cNames );

   for ( int cIdx=0; cIdx < cCount; cIdx++ ) {
      GarbageString *gs = new GarbageString( vm );
      gs->bufferize( cNames[cIdx] );

      ary->append( gs );

      free( cNames[cIdx] );
   }

   free( cNames );

   vm->retval( ary );
}

FALCON_FUNC DBIRecordset_getColumnCount( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   vm->retval( dbr->getColumnCount() );
}

FALCON_FUNC DBIRecordset_asString( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   String value;
   dbi_status retval = dbr->asString( columnIndexI->asInteger(), value );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil ();        // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_asInteger( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   int32 value;
   dbi_status retval = dbr->asInteger( columnIndexI->asInteger(), value );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil ();           // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_asInteger64( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   int64 value;
   dbi_status retval = dbr->asInteger64( columnIndexI->asInteger(), value );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil (); // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_asNumeric( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   numeric value;
   dbi_status retval = dbr->asNumeric( columnIndexI->asInteger(), value );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil(); // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_asDate( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   // create the timestamps
   TimeStamp *ts = new TimeStamp();
   Item *ts_class = vm->findGlobalItem( "TimeStamp" );
   //if we wrote the std module, can't be zero.
   fassert( ts_class != 0 );
   CoreObject *value = ts_class->asClass()->createInstance();
   dbi_status retval = dbr->asDate( columnIndexI->asInteger(), *ts );
   value->setUserData( ts );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil(); // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_asTime( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   // create the timestamps
   TimeStamp *ts = new TimeStamp();
   Item *ts_class = vm->findGlobalItem( "TimeStamp" );
   //if we wrote the std module, can't be zero.
   fassert( ts_class != 0 );
   CoreObject *value = ts_class->asClass()->createInstance();
   dbi_status retval = dbr->asTime( columnIndexI->asInteger(), *ts );
   value->setUserData( ts );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil(); // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_asDateTime( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   // create the timestamps
   TimeStamp *ts = new TimeStamp();
   Item *ts_class = vm->findGlobalItem( "TimeStamp" );
   //if we wrote the std module, can't be zero.
   fassert( ts_class != 0 );
   CoreObject *value = ts_class->asClass()->createInstance();
   dbi_status retval = dbr->asDateTime( columnIndexI->asInteger(), *ts );
   value->setUserData( ts );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil(); // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_getLastError( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   String value;
   dbi_status retval = dbr->getLastError( value );
   if ( retval != dbi_ok ) {
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                      .desc( "Could not get last error message " ) ) );
      return;
   }

   GarbageString *gs = new GarbageString( vm );
   gs->bufferize( value );

   vm->retval( gs );

   vm->retval( 0 );
}

FALCON_FUNC DBIRecordset_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   dbr->close();
}

//======================================================
// DBI error
//======================================================

FALCON_FUNC DBIError_init( VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new DBIError ) );

   ::Falcon::core::Error_init( vm );
}


}
}

/* end of dbi_ext.cpp */

