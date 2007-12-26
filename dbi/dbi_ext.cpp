/*
   FALCON - The Falcon Programming Language.
FILE: dbi_ext.cpp
   
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
   // todo: raise on error
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

FALCON_FUNC DBIRecordset_fetch( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   vm->retval( 0 );
}

FALCON_FUNC DBIRecordset_fetchColumns( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   CoreArray *ary = new CoreArray( vm, dbr->fetchColumnCount() );
   DBIRecordset::dbr_status retval;
   dbr->fetchColumns( ary );
   
   vm->retval( ary );
}

FALCON_FUNC DBIRecordset_fetchRowCount( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   vm->retval( dbr->fetchRowCount() );
}

FALCON_FUNC DBIRecordset_fetchColumnCount( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   vm->retval( dbr->fetchColumnCount() );
}

FALCON_FUNC DBIRecordset_getLastError( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   
   vm->retval( 0 );
}

}
}

/* end of dbi_ext.cpp */

