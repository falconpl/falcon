/*
 * FALCON - The Falcon Programming Language.
 * FILE: oracle_srv.cpp
 *
 * Oracle Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Steven Oliver
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <string.h>
#include <stdio.h>
#include <errmsg.h>

#include <falcon/engine.h>
#include "oracle_mod.h"

namespace Falcon
{
    /******************************************************************************
     * Recordset class
     *****************************************************************************/

    // FIXME
    DBIRecordsetOracle::DBIRecordsetOracle( DBIHandle *dbh, MYSQL_RES *res )
        : DBIRecordset( dbh )
    {
        o_res = res;

        o_row = -1; // BOF
        o_rowCount = mysql_num_rows( res ); // Only valid when using mysql_store_result instead of use_result
        o_columnCount = mysql_num_fields( res );
        o_fields = mysql_fetch_fields( res );
    }

    DBIRecordsetOracle::~DBIRecordsetOracle()
    {
        if ( o_res != NULL )
            close();
    }

    // FIXME
    dbi_type DBIRecordsetMySQL::getFalconType( int typ )
    {
        switch ( typ )
        {
            case MYSQL_TYPE_TINY:
            case MYSQL_TYPE_SHORT:
            case MYSQL_TYPE_LONG:
            case MYSQL_TYPE_INT24:
            case MYSQL_TYPE_BIT:
            case MYSQL_TYPE_YEAR:
                return dbit_integer;

            case MYSQL_TYPE_LONGLONG:
                return dbit_integer64;

            case MYSQL_TYPE_DECIMAL:
            case MYSQL_TYPE_NEWDECIMAL:
            case MYSQL_TYPE_FLOAT:
            case MYSQL_TYPE_DOUBLE:
                return dbit_numeric;

            case MYSQL_TYPE_DATE:
                return dbit_date;

            case MYSQL_TYPE_TIME:
                return dbit_time;

            case MYSQL_TYPE_DATETIME:
                return dbit_datetime;

            default:
                return dbit_string;
        }
    }

    // FIXME
    dbi_status DBIRecordsetOracle::next()
    {
        m_rowData = mysql_fetch_row( m_res );
        if ( m_rowData == NULL ) {
            return dbi_eof;
        } else if ( mysql_num_fields( m_res ) == 0 ) {
            unsigned int err = mysql_errno( ((DBIHandleMySQL *) m_dbh)->getConn() );
            switch ( err )
            {
                case CR_SERVER_LOST:
                    return dbi_invalid_connection;

                case CR_UNKNOWN_ERROR:
                    return dbi_error; // TODO: provide better error information

                default:
                    return dbi_error; // TODO: provide better error information
            }
        }

        m_row++;

        // Fetch lengths of each field so we can later deal with binary values that may contain ZERO
        // or NULL values right in the middle of a string.
        m_fieldLengths = mysql_fetch_lengths( m_res );

        return dbi_ok;
    }

    int DBIRecordsetOracle::getColumnCount()
    {
        return o_columnCount;
    }

    // FIXME
    dbi_status DBIRecordsetOracle::getColumnNames( char *names[] )
    {
        for ( int cIdx = 0; cIdx < m_columnCount; cIdx++ )
            names[cIdx] = m_fields[cIdx].name;

        return dbi_ok;
    }

    dbi_status DBIRecordsetOracle::getColumnTypes( dbi_type *types )
    {
        for ( int cIdx = 0; cIdx < m_columnCount; cIdx++ )
            types[cIdx] = getFalconType( m_fields[cIdx].type );

        return dbi_ok;
    }

    // FIXME
    dbi_status DBIRecordsetOracle::asString( const int columnIndex, String &value )
    {
        if ( columnIndex >= o_columnCount )
            return dbi_column_range_error;
        else if ( o_res == NULL )
            return dbi_invalid_recordset;
        else if ( o_rowData[columnIndex] == NULL )
            return dbi_nil_value;

        // TODO: check proper field encoding and transcode.
        value.fromUTF8( o_rowData[columnIndex] );
        return dbi_ok;
    }

    dbi_status DBIRecordsetOracle::asBlobID( const int columnIndex, String &value )
    {
        return dbi_not_implemented;
    }

    // FIXME
    dbi_status DBIRecordsetOracle::asBoolean( const int columnIndex, bool &value )
    {
        if ( columnIndex >= m_columnCount )
            return dbi_column_range_error;
        else if ( m_res == NULL )
            return dbi_invalid_recordset;
        else if ( m_rowData[columnIndex] == NULL )
            return dbi_nil_value;

        char *v = m_rowData[columnIndex];

        if (strncmp( v, "t", 1 ) == 0 || strncmp( v, "T", 1 ) == 0 || strncmp( v, "1", 1 ) == 0)
            value = true;
        else
            value = false;

        return dbi_ok;
    }

    // FIXME
    dbi_status DBIRecordsetOracle::asInteger( const int columnIndex, int32 &value )
    {
        if ( columnIndex >= m_columnCount )
            return dbi_column_range_error;
        else if ( m_res == NULL )
            return dbi_invalid_recordset;
        else if ( m_rowData[columnIndex] == NULL )
            return dbi_nil_value;

        value = atoi( m_rowData[columnIndex] );

        return dbi_ok;
    }

    // Fixme
    dbi_status DBIRecordsetOracle::asInteger64( const int columnIndex, int64 &value )
    {
        if ( columnIndex >= m_columnCount )
            return dbi_column_range_error;
        else if ( m_res == NULL )
            return dbi_invalid_recordset;
        else if ( m_rowData[columnIndex] == NULL )
            return dbi_nil_value;

        value = atoll( m_rowData[columnIndex] );

        return dbi_ok;
    }

    // FIXME
    dbi_status DBIRecordsetOracle::asNumeric( const int columnIndex, numeric &value )
    {
        if ( columnIndex >= m_columnCount )
            return dbi_column_range_error;
        else if ( m_res == NULL )
            return dbi_invalid_recordset;
        else if ( m_rowData[columnIndex] == NULL )
            return dbi_nil_value;

        value = atof( m_rowData[columnIndex] );

        return dbi_ok;
    }

    dbi_status DBIRecordsetOracle::asDate( const int columnIndex, TimeStamp &value )
    {
        if ( columnIndex >= m_columnCount )
            return dbi_column_range_error;
        else if ( m_res == NULL )
            return dbi_invalid_recordset;
        else if ( m_rowData[columnIndex] == NULL )
            return dbi_nil_value;

        String tv( m_rowData[columnIndex] );

        // 2007-12-27
        // 0123456789

        int64 year, month, day;
        tv.subString( 0, 4 ).parseInt( year );
        tv.subString( 5, 7 ).parseInt( month );
        tv.subString( 8, 10 ).parseInt( day );

        value.m_year = year;
        value.m_month = month;
        value.m_day = day;
        value.m_hour = 0;
        value.m_minute = 0;
        value.m_second = 0;
        value.m_msec = 0;

        return dbi_ok;
    }

    dbi_status DBIRecordsetOracle::asTime( const int columnIndex, TimeStamp &value )
    {
        if ( columnIndex >= m_columnCount )
            return dbi_column_range_error;
        else if ( m_res == NULL )
            return dbi_invalid_recordset;
        else if ( m_rowData[columnIndex] == NULL )
            return dbi_nil_value;

        String tv( m_rowData[columnIndex] );

        // 01:02:03
        // 01234567

        int64 hour, minute, second;
        tv.subString( 0, 2 ).parseInt( hour );
        tv.subString( 3, 5 ).parseInt( minute );
        tv.subString( 6, 8 ).parseInt( second );

        value.m_year = 0;
        value.m_month = 0;
        value.m_day = 0;
        value.m_hour = hour;
        value.m_minute = minute;
        value.m_second = second;
        value.m_msec = 0;

        return dbi_ok;
    }

    // FIXME
    dbi_status DBIRecordsetOracle::asDateTime( const int columnIndex, TimeStamp &value )
    {
        if ( columnIndex >= m_columnCount )
            return dbi_column_range_error;
        else if ( m_res == NULL )
            return dbi_invalid_recordset;
        else if ( m_rowData[columnIndex] == NULL )
            return dbi_nil_value;

        String tv( m_rowData[columnIndex] );

        // 2007-10-20 01:02:03
        // 0123456789012345678

        int64 year, month, day, hour, minute, second;
        tv.subString(  0,  4 ).parseInt( year );
        tv.subString(  5,  7 ).parseInt( month );
        tv.subString(  8, 10 ).parseInt( day );
        tv.subString( 11, 13 ).parseInt( hour );
        tv.subString( 14, 16 ).parseInt( minute );
        tv.subString( 17, 19 ).parseInt( second );

        value.m_year = year;
        value.m_month = month;
        value.m_day = day;
        value.m_hour = hour;
        value.m_minute = minute;
        value.m_second = second;
        value.m_msec = 0;

        return dbi_ok;
    }

    int DBIRecordsetOracle::getRowCount()
    {
        return o_rowCount;
    }

    int DBIRecordsetOracle::getRowIndex()
    {
        return o_row;
    }

    void DBIRecordsetOracle::close()
    {
        if ( m_res != NULL ) {
            mysql_free_result( m_res );
            m_res = NULL;
        }
    }

    // FIXME
    dbi_status DBIRecordsetMySQL::getLastError( String &description )
    {
        MYSQL *conn = ( (DBIHandleMySQL *) m_dbh )->getConn();


            return dbi_invalid_connection;

        const char *errorMessage = mysql_error( conn );

        if ( errorMessage == NULL )
            return dbi_no_error_message;

        description.bufferize( errorMessage );

        return dbi_ok;
    }

    /******************************************************************************
     * Transaction class
     *****************************************************************************/

    DBIStatementMySQL::DBITransactionOracle( DBIHandle *dbh )
        : DBIStatement( dbh )
    {
        o_inTransaction = false;
    }

    // FIXME
    DBIRecordset *DBIStatementMySQL::query( const String &query, int64 &affectedRows, dbi_status &retval )
    {
        retval = dbi_ok;

        AutoCString asQuery( query );
        MYSQL *conn = ((DBIHandleMySQL *) m_dbh)->getConn();

        if ( mysql_real_query( conn, asQuery.c_str(), asQuery.length() ) != 0 )
        {
            switch ( mysql_errno( conn ) )
            {
                case CR_COMMANDS_OUT_OF_SYNC:
                    retval = dbi_query_error;
                    break;

                case CR_SERVER_GONE_ERROR:
                case CR_SERVER_LOST:
                    retval = dbi_invalid_connection;
                    break;

                default:
                    retval = dbi_error;
            }
            return NULL;
        }

        if ( mysql_field_count( conn ) > 0 )
        {
            MYSQL_RES* res = mysql_store_result( conn );

            if ( res == NULL ) {
                retval = dbi_memory_allocation_error;
                return NULL;
            }

            return new DBIRecordsetOracle( m_dbh, res );
        }

        affectedRows = (int64) mysql_affected_rows( conn );

        // query without recordset
        return NULL;
    }

    // FIXME
    dbi_status DBITransactionOracle::begin()
    {
        dbi_status retval;
        int64 dummy;
        query( "BEGIN", dummy, retval );

        if ( retval == dbi_ok )
            m_inTransaction = true;

        return retval;
    }

    dbi_status DBITransactionOracle::commit()
    {
        dbi_status retval;
        int64 dummy;
        Connection::commit();

        o_inTransaction = false;

        return retval;
    }

    dbi_status DBITransactionOracle::rollback()
    {
        dbi_status retval;
        int64 dummy;
        Connection::rollback();

        o_inTransaction = false;

        return retval;
    }

    dbi_status DBITransactionOracle::close()
    {
        // TODO return a status code here because of the potential commit
        return o_dbh->closeTransaction( this );
    }

    // FIXME 
    dbi_status DBITransactionOracle::getLastError( String &description )
    {
        MYSQL *conn = static_cast<DBIHandleMySQL *>( m_dbh )->getConn();

        if ( conn == NULL )
            return dbi_invalid_connection;

        const char *errorMessage = mysql_error( conn );

        if ( errorMessage == NULL )
            return dbi_no_error_message;

        description.bufferize( errorMessage );

        return dbi_ok;
    }


    DBIBlobStream *DBITransactionOracle::openBlob( const String &blobId, dbi_status &status )
    {
        status = dbi_not_implemented;
        return 0;
    }

    DBIBlobStream *DBITransactionOracle::createBlob( dbi_status &status, const String &params,
            bool bBinary )
    {
        status = dbi_not_implemented;
        return 0;
    }

    /******************************************************************************
     * DB Handler class
     *****************************************************************************/
    DBIHandleOracle::~DBIHandleOracle()
    {
        DBIHandleOracle::close();
    }

    DBIStatement *DBIHandleOracle::startTransaction()
    {
        DBITransactionOracle *t = new DBITransactionOracle( this );
        if ( t->begin() != dbi_ok ) {
            // TODO: filter useful information to the script level
            delete t;

            return NULL;
        }

        return t;
    }

    DBIStatement* DBIHandleOracle::getDefaultTransaction()
    {
        return o_connTr == NULL ? (o_connTr = new DBITransactionOracle( this )): o_connTr;
    }

    DBIHandleMySQL::DBIHandleOracle()
    {
        o_conn = NULL;
        o_connTr = NULL;
    }

    DBIHandleMySQL::DBIHandleOracle( Connection *conn )
    {
        o_conn = conn;
        o_connTr = NULL;
    }

    dbi_status DBIHandleOracle::closeTransaction( DBIStatement *tr )
    {
        return dbi_ok;
    }

    int64 DBIHandleOracle::getLastInsertedId()
    {
        return oracle_insert_id( o_conn );
    }

    int64 DBIHandleOracle::getLastInsertedId( const String& sequenceName )
    {
        return oracle_insert_id( o_conn );
    }

    dbi_status DBIHandleOracle::getLastError( String &description )
    {
        if ( o_conn == NULL )
            return dbi_invalid_connection;

        const char *errorMessage = getMessage();
        if ( errorMessage == NULL )
            return dbi_no_error_message;

        description.bufferize( errorMessage );

        return dbi_ok;
    }

    dbi_status DBIHandleOracle::escapeString( const String &value, String &escaped )
    {
        if ( value.length() == 0 )
            return dbi_ok;

        AutoCString asValue( value );

        int maxLen = ( value.length() * 2 ) + 1;
        char *cTo = (char *) memAlloc( sizeof( char ) * maxLen );

        // FIXME
        size_t convertedSize = mysql_real_escape_string( o_conn, cTo,
                asValue.c_str(), asValue.length() );

        if ( convertedSize < value.length() ) {
            memFree( cTo );
            return dbi_error;
        }

        escaped.fromUTF8( cTo );
        memFree( cTo );

        return dbi_ok;
    }

    dbi_status DBIHandleOracle::close()
    {
        // Make sure we have something to close
        if ( o_conn != NULL ) {
            // Kill the connection then the environment
            env->terminateConnection(o_conn);
            Environment::terminateEnvironment(env);
        }

        return dbi_ok;
    }

    /******************************************************************************
     * Main service class
     ******************************************************************************/

    dbi_status DBIServiceOracle::init()
    {
        return dbi_ok;
    }

    DBIHandle *DBIServiceOracle::connect( const String &parameters, bool persistent,
            dbi_status &retval, String &errorMessage )
    {
        char *host, *user, *passwd, *db, *port, *unixSocket, *clientFlags;
        unsigned int iPort, iClientFlag;
        char** vals[] = { &host, &user, &passwd, &db, &port, &unixSocket, &clientFlags, 0 };

        Environment *env;
        Connection *conn;

        AutoCString asConnParams( parameters );
        char* connp = (char*) asConnParams.c_str();
        char*** elem = vals;
        char* beg = connp;

        // tokenize ","
        while( *connp && *elem )
        {
            if ( *connp == ',' )
            {
                if ( beg == connp )
                {
                    // empty "," ?
                    **elem = 0;
                }
                else
                    **elem = beg;

                ++elem;
                *connp = '\0';
                ++connp;
                beg = connp;
            }
            else
                ++connp;
        }

        // get the last element
        if ( *elem )
        {
            **elem = beg;
            ++elem;
            // nulls remaining parameters.
            while( *elem )
            {
                **elem = 0;
                ++elem;
            }
        }

        env = Environment::createEnvironment();

        // We'll need a name and password at the very least here.
        // TODO Fill in the other parameters
        conn = env->createConnection(user, passwd);

        if ( conn == NULL )
        {
            errorMessage = getMessage();
            errorMessage.bufferize();
            terminateConnection( conn );
            terminateEnvironment( env );
            retval = dbi_connect_error;
            return NULL;
        }
        else
        {
#if (OCCI_MAJOR_VERSION > 9)
            env->setCacheSortedFlush(true);
#endif
            retval = dbi_ok;
            return new DBIHandleOracle( conn );
        }

    }

    CoreObject *DBIServiceOracle::makeInstance( VMachine *vm, DBIHandle *dbh )
    {
        Item *cl = vm->findWKI( "Oracle" );
        if ( cl == 0 || ! cl->isClass() || cl->asClass()->symbol()->name() != "Oracle" )
        {
            throw new DBIError( ErrorParam( dbi_driver_not_found, __LINE__ )
                    .desc( "Oracle DBI driver was not found" ) );
            return 0;
        }

        CoreObject *obj = cl->asClass()->createInstance();
        obj->setUserData( dbh );

        return obj;
    }
}/* namespace Falcon */

/* end of oracle_srv.cpp */

