/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_mod.cpp
 *
 *
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Mon, 13 Apr 2009 18:56:48 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */


#include "dbi_mod.h"
#include "dbi.h"
#include <dbiservice.h>

/******************************************************************************
 * Local Helper Functions - DBH database handle
 *****************************************************************************/
namespace Falcon
{

DBIRecordset *dbh_baseQueryOne( VMachine *vm, int startAt )
{
   CoreObject *self = vm->self().asObject();
   
   DBIItemBase *dbiitem = static_cast<DBIItemBase *>( self->getUserData() );
   DBIHandle* dbh;
   DBITransaction* dbt;
   
   if ( dbiitem->isHandle() )
   {
      dbh = static_cast<DBIHandle*>(dbiitem);
      dbt = dbh->getDefaultTransaction();
   }
   else {
      dbt = static_cast<DBITransaction*>(dbiitem);
      dbh = dbt->getHandle();
   }

   String sql;
   dbh_realSqlExpand( vm, dbh, sql, startAt );

   dbi_status retval;
   int64 affected;
   DBIRecordset *recSet = dbt->query( sql, affected, retval );
   if ( recSet == NULL ) {
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_no_results, __LINE__ )
            .desc( "No results from query")
            .extra( "queryOne" ) );
   }

   if ( retval != dbi_ok ) {
      String errorMessage;
      dbt->getLastError( errorMessage );

      throw  new DBIError( ErrorParam( DBI_ERROR_BASE + retval, __LINE__ )
                                       .desc( errorMessage ) );
   }

   dbi_status nextStatus = recSet->next();
   if ( nextStatus != dbi_ok ) {
      vm->retnil();
      return NULL;
   }

   return recSet;
}

DBIRecordset *dbh_query_base( DBITransaction* dbt, const String &sql )
{
   dbi_status retval;
   int64 affected;
   DBIRecordset *recSet = dbt->query( sql, affected, retval );

   if ( retval != dbi_ok ) {
      String errorMessage;
      dbt->getLastError( errorMessage );

      throw new DBIError( ErrorParam( DBI_ERROR_BASE + retval, __LINE__ )
                                       .desc( errorMessage ) );
   }

   return recSet;
}


dbi_type *recordset_getTypes( DBIRecordset *recSet )
{
   if (recSet == NULL )
      return NULL;
   dbi_type *cTypes = (dbi_type *) memAlloc( sizeof( dbi_type ) * recSet->getColumnCount() );
   recSet->getColumnTypes( cTypes );
   return cTypes;
}


int dbh_itemToSqlValue( DBIHandle *dbh, const Item *i, String &value )
{
   switch( i->type() ) {
      case FLC_ITEM_BOOL:
         value = i->asBoolean() ? "TRUE" : "FALSE";
         return 1;

      case FLC_ITEM_INT:
         value.writeNumber( i->asInteger() );
         return 1;

      case FLC_ITEM_NUM:
         value.writeNumber( i->asNumeric(), "%f" );
         return 1;

      case FLC_ITEM_STRING:
         dbh->escapeString( *i->asString(), value );
         value.prepend( "'" );
         value.append( "'" );
         return 1;

      case FLC_ITEM_OBJECT: {
            CoreObject *o = i->asObject();
            //vm->itemToString( value, ??? )
            if ( o->derivedFrom( "TimeStamp" ) ) {
               TimeStamp *ts = (TimeStamp *) o->getUserData();
               ts->toString( value );
               value.prepend( "'" );
               value.append( "'" );
               return 1;
            }
            return 0;
         }

      case FLC_ITEM_NIL:
         value = "NULL";
         return 1;

      default:
         return 0;
   }
}


int dbh_realSqlExpand( VMachine *vm, DBIHandle *dbh, String &sql, int startAt )
{
   String errorMessage;

   Item *sqlI = vm->param( startAt );
   if ( sqlI == 0 || ! sqlI->isString() ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S") );
      return 0;
   }

   sql = *sqlI->asString();
   sql.bufferize();

   startAt++;

   uint32 colonPos = sql.find( ":", 0 );

   if ( colonPos != csh::npos )
   {
      // Check param 'startAt', if a dict or object, we treat things special
      CoreDict *dict = NULL;
      CoreObject *obj = NULL;

      Item *psI = vm->param( startAt );
      if ( psI != 0 ) {
         if ( psI->isDict() )
            dict = psI->asDict();
         else if ( psI->isObject() )
            obj = psI->asObject();
      }

      while ( colonPos != csh::npos )
      {
         Item *i = 0;
         Item i_dummy;
         int colonSize = 1;

         if ( colonPos == sql.length() - 1 )
         {
            throw  new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_sql_expand_error,
                                                        __LINE__ )
                     .desc( "Stray : charater at the end of query" ) );
            return 0;
         }
         else
         {
            if ( sql.getCharAt( colonPos + 1 ) == ':' ) {
               sql.remove( colonPos, 1 );
               colonPos = sql.find( ":", colonPos + 1 );
               continue;
            }

            uint32 commaPos = sql.find( ",", colonPos );
            uint32 spacePos = sql.find( " ", colonPos );
            uint32 ePos = commaPos < spacePos ? commaPos : spacePos;

            if ( dict != NULL || obj != NULL )
            {
               String word = sql.subString( colonPos + 1, ePos );
               if ( dict != NULL ) {
                  // Reading from the dict
                  Item wordI( &word );
                  i = dict->find( wordI );
               } else {
                  // Must be obj
                  i = obj->getProperty( word, i_dummy ) ? &i_dummy : 0;
               }

               if ( i == 0 ) {
                  throw  new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_sql_expand_error, __LINE__ )
                             .desc( "Word expansion was not found in dictionary/object" )
                             .extra( word ) );
                  return 0;
               }

               colonSize += word.length();
            }
            else
            {
               AutoCString asTmp( sql.subString( colonPos + 1, ePos ) );
               int pIdx = atoi( asTmp.c_str() );

               if ( pIdx == 0 ) {
                  throw  new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_sql_expand_error,
                                                              __LINE__ )
                          .desc( "Failed to parse colon expansion" )
                          .extra( "from: " + sql.subString( colonPos ) ) );
                  return 0;
               }

               if ( pIdx > 99 ) colonSize++; // it is 3 digits !?!?
               if ( pIdx > 9 ) colonSize++;  // it is 2 digits
               colonSize++;                  // it exists
               i = vm->param( pIdx + ( startAt - 1 ) );

               if ( i == 0 )
               {
                  errorMessage.writeNumber( (int64) pIdx );
                  throw  new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_sql_expand_error, __LINE__ )
                       .desc("Positional expansion out of range")
                       .extra( errorMessage ) );
                  return 0;
               }
            }
         }

         String value;
         if ( dbh_itemToSqlValue( dbh, i, value ) == 0 )
         {
            throw  new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_sql_expand_type_error, __LINE__ )
                     .desc( "Failed to expand a value due to it being an unknown type" )
                     .extra( "from: " + sql.subString( colonPos ) ) );
            return 0;
         }

         sql.insert( colonPos, colonSize, value );
         colonPos += value.length();
         colonPos = sql.find( ":", colonPos );
      }
   }

   return 1;
}


void dbh_return_recordset( VMachine *vm, DBIRecordset *rec )
{
   Item *rsclass = vm->findWKI( "%DBIRecordset" );
   fassert( rsclass != 0 && rsclass->isClass() );

   CoreObject *oth = rsclass->asClass()->createInstance();
   oth->setUserData( rec );
   vm->retval( oth );
}

/******************************************************************************
 * Local Helper Functions - DBR recordset handle
 *****************************************************************************/

int dbr_getItem( VMachine *vm, DBIRecordset *dbr, dbi_type typ, int cIdx, Item &item )
{
   switch ( typ )
   {
      case dbit_string: {
         String value;
         dbi_status retval = dbr->asString( cIdx, value );
         switch ( retval )
         {
            case dbi_ok: {
               CoreString *gsValue = new CoreString;
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

      case dbit_integer: {
         int32 value;
         if ( dbr->asInteger( cIdx, value ) != dbi_nil_value )
            item.setInteger( (int64) value );
      }
      break;

      case dbit_boolean: {
         bool value;
         if ( dbr->asBoolean( cIdx, value ) != dbi_nil_value ) {
            item.setBoolean( value );
         }
      }

      case dbit_integer64: {
         int64 value;
         if ( dbr->asInteger64( cIdx, value ) != dbi_nil_value )
            item.setInteger( value );
      }
      break;

      case dbit_numeric: {
         numeric value;
         if ( dbr->asNumeric( cIdx, value ) != dbi_nil_value )
            item.setNumeric( value );
      }
      break;

      case dbit_date: {
         TimeStamp *ts = new TimeStamp();
         if ( dbr->asDate( cIdx, *ts ) != dbi_nil_value ) {
            Item *ts_class = vm->findWKI( "TimeStamp" );
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            value->setUserData( ts );
            item.setObject( value );
         }
      }
      break;

      case dbit_time: {
         TimeStamp *ts = new TimeStamp();
         if ( dbr->asTime( cIdx, *ts ) != dbi_nil_value ) {
            Item *ts_class = vm->findWKI( "TimeStamp" );
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            value->setUserData( ts );
            item.setObject( value );
         }
      }
      break;

      case dbit_datetime: {
         TimeStamp *ts = new TimeStamp();
         if ( dbr->asDateTime( cIdx, *ts ) != dbi_nil_value ) {
            Item *ts_class = vm->findWKI( "TimeStamp" );
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

int dbr_checkValidColumn( VMachine *vm, DBIRecordset *dbr, int cIdx )
{
   if ( cIdx >= dbr->getColumnCount() ) {
      String errorMessage = "Column index (";
      errorMessage.writeNumber( (int64) cIdx );
      errorMessage += ") is out of range";

      throw new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_column_range_error, __LINE__ )
                                      .desc( errorMessage ) );
      return 0;
   } else if ( dbr->getRowIndex() == -1 ) {
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_row_index_invalid, __LINE__ )
                                      .desc( "Invalid current row index" ) );
      return 0;
   }

   return 1;
}


void dbr_execute( VMachine *vm, DBIHandle *dbh, const String &sql )
{
   dbi_status retval;
   int64 affectedRows;
   DBIRecordset* rs;
   
   DBITransaction* tr;
   
   if ( vm->paramCount() == 0 )
      tr = dbh->getDefaultTransaction();
   else {
      Item *trI = vm->param( 0 );
      if ( trI == 0 || ! trI->isObject() || ! trI->asObject()->derivedFrom( "DBITransaction" ) )
      {
         throw  new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( "DBITransaction" ) );
         return;
      }
      CoreObject *trO = trI->asObject();
      tr = static_cast<DBITransaction *>( trO->getUserData() );
   }
   
   rs = tr->query( sql, affectedRows, retval );
   
   // we don't expect a return recordset
   #ifndef NDEBUG   
      fassert( rs == 0 );
   #else
      delete rs; // can be 0
   #endif
      
   if ( retval == dbi_ok )
      vm->retval( affectedRows );
   else {
      String errorMessage;
      tr->getLastError( errorMessage );
      throw  new DBIError( ErrorParam( DBI_ERROR_BASE + retval, __LINE__ )
               .desc( errorMessage ) );
   }
}

int dbr_getPersistPropertyNames( VMachine *vm, CoreObject *self, String columnNames[], int maxColumnCount )
{
   Item persistI;

   if ( ! self->getProperty( "_persist", persistI ) || persistI.isNil() ) {
      // No _persist, loop through all public properties
      const PropertyTable &pt = self->generator()->properties();
      int pCount = pt.size();
      int cIdx = 0;

      for ( int pIdx=0; pIdx < pCount; pIdx++ ) {
         const String &p = *pt.getEntry( pIdx ).m_name;
         if ( p.getCharAt( 0 ) != '_' ) {
            Item i;
            self->getProperty( p, i );
            if ( i.isInteger() || i.isNumeric() || i.isObject() || i.isString() ) {
               columnNames[cIdx] = p;
               cIdx++;
            }
         }
      }

      return cIdx;
   } else if ( ! persistI.isArray() ) {
      // They gave a _persist property, but it's not an array
      throw  new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( "_persist.type()!=A" ) );
      return 0;
   } else {
      // They gave a _persist property, trust it
      CoreArray *persist = persistI.asArray();
      int cCount = persist->length();

      for ( int cIdx=0; cIdx < cCount; cIdx++) {
         const Item &pi = persist->at( cIdx );
         if ( ! pi.isString() )
         {
            throw  new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_row_index_invalid, __LINE__ )
                     .desc( "There was a non-string item in the \"_persist\" property" ) );
            return 0;
         }
         else
            columnNames[cIdx] = *pi.asString();
      }

      return cCount;
   }
}

}

/* end of dbi_mod.cpp */
