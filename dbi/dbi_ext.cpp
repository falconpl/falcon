/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_ext.cpp
   
   DBI Falcon extension interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai, Jeremy Cowgar
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
   if ( provider != 0 ) 
   {
      // if it's 0, the service has already raised an error in the vm and we have nothing to do.
      String connectErrorMessage;
      DBIService::dbi_status status;
      DBIHandle *hand = provider->connect( connString, false, status, connectErrorMessage );
      if ( hand == 0 ) 
      {
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
   if ( trans == 0 )
   {
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
   if ( sqlI == 0 || ! sqlI->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }
   
   DBITransaction::dbt_status retval;
   DBIRecordset *recSet = dbh->query( *sqlI->asString(), retval );
   
   if ( retval != DBITransaction::s_ok )
   {
      // TODO: supply the real error message
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                       .desc( "Error processing SQL" ) ) );
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
   
   DBITransaction::dbt_status retval;
   int affectedRows = dbh->execute( *sqlI->asString(), retval );
   
   if ( retval != DBITransaction::s_ok )
   {
      // TODO: report real error here
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                       .desc( "Error executing query" ) ) );
   }
   
   vm->retval( affectedRows );
}

FALCON_FUNC DBIHandle_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   dbh->close();
   // TODO: raise on error
}

FALCON_FUNC DBIHandle_sqlExpand( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   
   Item *sqlI = vm->param( 0 );
   String *sql = sqlI->asString();
   
   uint32 lastMarkPos = 0;
   
   for ( int paramIdx = 1; paramIdx < vm->paramCount(); paramIdx++ )
   {
      uint32 markPos = sql->find( "?", lastMarkPos );
      
      if ( markPos == csh::npos )
         break;  // TODO: throw an error, don't exit silently
      
      if ( markPos < sql->length() && sql->getCharAt( markPos + 1 ) == '?' )
      {
         sql->remove( markPos, 1 );
         
         markPos++;
         markPos = sql->find( "?", markPos );
         if ( markPos == csh::npos )
            break; // TODO: throw an error, don't exit silently
      }
      
      // Convert based on type
      
      Item *i = vm->param( paramIdx );
      String s;
      
      if ( i->isInteger() )
         s.writeNumber( i->asInteger() );
      else if ( i->isNumeric() )
         s.writeNumber( i->asNumeric(), "%f");
      else if ( i->isString() )
      {
         dbh->escapeString( *i->asString(), s );
         s.prepend("'");
         s.append("'");
      }
      
      sql->insert( markPos, 1, s );
      
      lastMarkPos = markPos;
   }
   
   GarbageString *gs = new GarbageString( vm );
   gs->bufferize( *sql );
   
   vm->retval( gs );
}

/**********************************************************
   Transaction class
 **********************************************************/

FALCON_FUNC DBITransaction_query( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );
   
   Item *i_query = vm->param(0);
   if( i_query == 0 || ! i_query->isString() )
   {
      // raise error
      return;
   }
   
   /*
      if ( dbt->query( *i_query->asString() ) != DBITransaction::s_ok )
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

FALCON_FUNC DBIRecordset_fetchArray( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   DBIRecordset::dbr_status nextRetVal = dbr->next();   
   switch ( nextRetVal )
   {
   case DBIRecordset::s_ok:
      break;
      
   case DBIRecordset::s_eof:
      vm->retnil();
      return ;
      
   default:
      // TODO: Handle error
      break;
   }
   
   int cCount = dbr->getColumnCount();
   CoreArray *ary = new CoreArray( vm, cCount );
   CoreArray *cTypes = new CoreArray( vm, cCount );
   dbr->getColumnTypes( cTypes );
   
   for ( int cIdx = 0; cIdx < cCount; cIdx++ )
   {
      DBIRecordset::dbr_status retval;
      Item *i;
      
      switch ( cTypes->at( cIdx ).asInteger() )
      {
      case dbit_string:
         {
            String value;
            retval = dbr->asString( cIdx, value );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               ary->append( k );
            }
            else if ( retval == DBIRecordset::s_ok )
            {
               GarbageString *gsValue = new GarbageString( vm );
               gsValue->bufferize( value );
               ary->append( gsValue );
            }
            else
            {
               // TODO: handle error
            }
         }
         break;
         
      case dbit_integer:
         {
            int32 value;
            retval = dbr->asInteger( cIdx, value );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               ary->append( k );
            }
            else
            {
               ary->append( (int64) value );
            }
         }
         break;
         
      case dbit_integer64:
         {
            int64 value;
            retval = dbr->asInteger64( cIdx, value );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               ary->append( k );
            }
            else
            {
               ary->append( value );
            }
         }
         break;
         
      case dbit_numeric:
         {
            numeric value;
            retval = dbr->asNumeric( cIdx, value );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               ary->append( k );
            }
            else
            {
               ary->append( value );
            }
         }
         break;
         
      case dbit_date:
         {
            // create the timestamps
            TimeStamp *ts = new TimeStamp();
            Item *ts_class = vm->findGlobalItem( "TimeStamp" );
            //if we wrote the std module, can't be zero.
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            DBIRecordset::dbr_status retval = dbr->asDate( cIdx, *ts );
            value->setUserData( ts );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               ary->append( k );
            }
            else
            {
               ary->append( value );
            }
         }
         break;
         
      case dbit_time:
         {
            // create the timestamps
            TimeStamp *ts = new TimeStamp();
            Item *ts_class = vm->findGlobalItem( "TimeStamp" );
            //if we wrote the std module, can't be zero.
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            DBIRecordset::dbr_status retval = dbr->asTime( cIdx, *ts );
            value->setUserData( ts );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               ary->append( k );
            }
            else
            {
               ary->append( value );
            }
         }
         break;
         
      case dbit_datetime:
         {
            // create the timestamps
            TimeStamp *ts = new TimeStamp();
            Item *ts_class = vm->findGlobalItem( "TimeStamp" );
            //if we wrote the std module, can't be zero.
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            DBIRecordset::dbr_status retval = dbr->asDateTime( cIdx, *ts );
            value->setUserData( ts );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               ary->append( k );
            }
            else
            {
               ary->append( value );
            }
         }
         break;
      }
   }
   
   vm->retval( ary );
}

FALCON_FUNC DBIRecordset_fetchDict( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   DBIRecordset::dbr_status nextRetVal = dbr->next();   
   switch ( nextRetVal )
   {
   case DBIRecordset::s_ok:
      break;
      
   case DBIRecordset::s_eof:
      vm->retnil();
      return ;
      
   default:
      // TODO: Handle error
      break;
   }
   
   int cCount = dbr->getColumnCount();
   CoreDict *dict = new PageDict( vm, cCount );
   CoreArray *cTypes = new CoreArray( vm, cCount );
   CoreArray *cNames = new CoreArray( vm, cCount );
   dbr->getColumnTypes( cTypes );
   dbr->getColumnNames( cNames );
   
   for ( int cIdx = 0; cIdx < cCount; cIdx++ )
   {
      DBIRecordset::dbr_status retval;
      GarbageString *gsName = new GarbageString( vm );
      gsName->bufferize( *cNames->at( cIdx ).asString() );
      
      Item *i;
      
      switch ( cTypes->at( cIdx ).asInteger() )
      {
      case dbit_string:
         {
            String value;
            retval = dbr->asString( cIdx, value );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               dict->insert( gsName, k );
            }
            else if ( retval == DBIRecordset::s_ok )
            {
               GarbageString *gsValue = new GarbageString( vm );
               gsValue->bufferize( value );
               dict->insert( gsName, gsValue );
            }
            else
            {
               // TODO: handle error
            }
         }
         break;
         
      case dbit_integer:
         {
            int32 value;
            retval = dbr->asInteger( cIdx, value );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               dict->insert( gsName, k );
            }
            else
            {
               dict->insert( gsName, (int64) value );
            }
         }
         break;
         
      case dbit_integer64:
         {
            int64 value;
            retval = dbr->asInteger64( cIdx, value );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               dict->insert( gsName, k );
            }
            else
            {
               dict->insert( gsName, (int64) value );
            }
         }
         break;
         
      case dbit_numeric:
         {
            numeric value;
            retval = dbr->asNumeric( cIdx, value );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               dict->insert( gsName, k );
            }
            else
            {
               dict->insert( gsName, (numeric) value );
            }
         }
         break;
      
         
      case dbit_date:
         {
            // create the timestamps
            Item *ts_class = vm->findGlobalItem( "TimeStamp" );
            TimeStamp *ts = new TimeStamp();
            //if we wrote the std module, can't be zero.
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            DBIRecordset::dbr_status retval = dbr->asDate( cIdx, *ts );
            value->setUserData( ts );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               dict->insert( gsName, k );
            }
            else
            {
               dict->insert( gsName, value );
            }
         }
         break;
         
      case dbit_time:
         {
            // create the timestamps
            TimeStamp *ts = new TimeStamp();
            Item *ts_class = vm->findGlobalItem( "TimeStamp" );
            //if we wrote the std module, can't be zero.
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            DBIRecordset::dbr_status retval = dbr->asTime( cIdx, *ts );
            value->setUserData( ts );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               dict->insert( gsName, k );
            }
            else
            {
               dict->insert( gsName, value );
            }
         }
         break;
         
      case dbit_datetime:
         {
            // create the timestamps
            TimeStamp *ts = new TimeStamp();
            Item *ts_class = vm->findGlobalItem( "TimeStamp" );
            //if we wrote the std module, can't be zero.
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            DBIRecordset::dbr_status retval = dbr->asDateTime( cIdx, *ts );
            value->setUserData( ts );
            
            if ( retval == DBIRecordset::s_nil_value )
            {
               Item k;
               dict->insert( gsName, k );
            }
            else
            {
               dict->insert( gsName, value );
            }
         }
         break;
      }
   }
   
   vm->retval( dict );
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
   
   CoreArray *ary = new CoreArray( vm, dbr->getColumnCount() );
   DBIRecordset::dbr_status retval;
   dbr->getColumnTypes( ary );
   
   vm->retval( ary );
}   

FALCON_FUNC DBIRecordset_getColumnNames( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   CoreArray *ary = new CoreArray( vm, dbr->getColumnCount() );
   DBIRecordset::dbr_status retval;
   dbr->getColumnNames( ary );
   
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
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }
   
   String value;
   DBIRecordset::dbr_status retval = dbr->asString( columnIndexI->asInteger(), value );
   
   if ( retval == DBIRecordset::s_nil_value )
   {
      vm->retnil();
   }
   else if ( retval != DBIRecordset::s_ok )
   {
      // TODO: handle the error
      vm->retnil ();
   }
   else
   {
      vm->retval( value );
   }
}

FALCON_FUNC DBIRecordset_asInteger( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }
   
   int32 value;
   DBIRecordset::dbr_status retval = dbr->asInteger( columnIndexI->asInteger(), value );
   
   if ( retval == DBIRecordset::s_nil_value )
   {
      vm->retnil();
   }
   else if ( retval != DBIRecordset::s_ok )
   {
      // TODO: handle the error
      vm->retnil ();
   }
   else
   {
      vm->retval( value );
   }
}

FALCON_FUNC DBIRecordset_asInteger64( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }
   
   int64 value;
   DBIRecordset::dbr_status retval = dbr->asInteger64( columnIndexI->asInteger(), value );
   
   if ( retval == DBIRecordset::s_nil_value )
   {
      vm->retnil();
   }
   else if ( retval != DBIRecordset::s_ok )
   {
      // TODO: handle the error
      vm->retnil ();
   }
   else
   {
      vm->retval( value );
   }
}

FALCON_FUNC DBIRecordset_asNumeric( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }
   
   numeric value;
   DBIRecordset::dbr_status retval = dbr->asNumeric( columnIndexI->asInteger(), value );
   
   if ( retval == DBIRecordset::s_nil_value )
   {
      vm->retnil();
   }
   else if ( retval != DBIRecordset::s_ok )
   {
      // TODO: handle the error
      vm->retnil ();
   }
   else
   {
      vm->retval( value );
   }
}

FALCON_FUNC DBIRecordset_asDate( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() )
   {
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
   DBIRecordset::dbr_status retval = dbr->asDate( columnIndexI->asInteger(), *ts );
   value->setUserData( ts );
   
   if ( retval == DBIRecordset::s_nil_value )
   {
      vm->retnil();
   }
   else if ( retval != DBIRecordset::s_ok )
   {
      // TODO: handle the error
      vm->retnil ();
   }
   else
   {
      vm->retval( value );
   }
}

FALCON_FUNC DBIRecordset_asTime( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() )
   {
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
   DBIRecordset::dbr_status retval = dbr->asTime( columnIndexI->asInteger(), *ts );
   value->setUserData( ts );
   
   if ( retval == DBIRecordset::s_nil_value )
   {
      vm->retnil();
   }
   else if ( retval != DBIRecordset::s_ok )
   {
      // TODO: handle the error
      vm->retnil ();
   }
   else
   {
      vm->retval( value );
   }
}

FALCON_FUNC DBIRecordset_asDateTime( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() )
   {
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
   DBIRecordset::dbr_status retval = dbr->asDateTime( columnIndexI->asInteger(), *ts );
   value->setUserData( ts );
   
   if ( retval == DBIRecordset::s_nil_value )
   {
      vm->retnil();
   }
   else if ( retval != DBIRecordset::s_ok )
   {
      // TODO: handle the error
      vm->retnil ();
   }
   else
   {
      vm->retval( value );
   }
}

FALCON_FUNC DBIRecordset_getLastError( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   vm->retval( 0 );
}

FALCON_FUNC DBIRecordset_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   dbr->close();
}

}
}

/* end of dbi_ext.cpp */

