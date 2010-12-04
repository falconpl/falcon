/*
 * FALCON - The Falcon Programming Language.
 * FILE: oracle_mod.h
 *
 * Oracle driver main module interface
 * -------------------------------------------------------------------
 * Author: Steven Oliver
 * Based on equivalent MySQL drivers by Jeremy Cowgar
 * -------------------------------------------------------------------
 * (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef DBI_ORACLE_H
#define DBI_ORACLE_H

#include "../include/dbiservice.h"

#include <occi.h>

namespace Falcon
{
    class DBIRecordsetOracle : public DBIRecordset
    {
        protected:
            int o_row;
            int o_rowCount;
            int o_columnCount;

            ResultSet *ORACLE_RES;

            unsigned long *o_fieldLengths;

            static dbi_type getFalconType( int typ );

        public:
            DBIRecordsetOracle( DBIHandle *dbh, ORACLE_RES *res );
            ~DBIRecordsetOracle();

            virtual dbi_status next();
            virtual int getRowCount();
            virtual int getRowIndex();
            virtual int getColumnCount();
            virtual dbi_status getColumnNames( char *names[] );
            virtual dbi_status getColumnTypes( dbi_type *types );
            virtual dbi_status asString( const int columnIndex, String &value );
            virtual dbi_status asBoolean( const int columnIndex, bool &value );
            virtual dbi_status asInteger( const int columnIndex, int32 &value );
            virtual dbi_status asInteger64( const int columnIndex, int64 &value );
            virtual dbi_status asNumeric( const int columnIndex, numeric &value );
            virtual dbi_status asDate( const int columnIndex, TimeStamp &value );
            virtual dbi_status asTime( const int columnIndex, TimeStamp &value );
            virtual dbi_status asDateTime( const int columnIndex, TimeStamp &value );
            virtual dbi_status asBlobID( const int columnIndex, String &value );
            virtual void close();
            virtual dbi_status getLastError( String &description );
    };


    class DBITransactionOracle : public DBIStatement
    {
        protected:
            bool o_inTransaction;

        public:
            DBITransactionOracle( DBIHandle *dbh );

            virtual DBIRecordset *query( const String &query, int64 &affected_rows, dbi_status &retval );
            virtual dbi_status begin();
            virtual dbi_status commit();
            virtual dbi_status rollback();
            virtual dbi_status close();
            virtual dbi_status getLastError( String &description );

            virtual DBIBlobStream *openBlob( const String &blobId, dbi_status &status );
            virtual DBIBlobStream *createBlob( dbi_status &status, const String &params= "",
                    bool bBinary = false );
    };


    class DBIHandleOracle : public DBIHandle
    {
        protected:
            Environment *o_env;
            Connection *o_conn;

            DBITransactionORACLE *m_connTr;

        public:
            DBIHandleOracle();
            DBIHandleOracle( ORALCE *conn );
            virtual ~DBIHandleOracle();

            Connection *getConn() { return o_conn; }

            virtual DBIStatement *startTransaction();
            virtual dbi_status closeTransaction( DBIStatement *tr );
            virtual int64 getLastInsertedId();
            virtual int64 getLastInsertedId( const String &value );
            virtual dbi_status getLastError( String &description );
            virtual dbi_status escapeString( const String &value, String &escaped );
            virtual dbi_status close();
            virtual DBIStatement* getDefaultTransaction();
    };

    class DBIServiceOracle : public DBIService
    {
        public:
            DBIServiceOracle() : DBIService( "DBI_oracle" ) {}

            virtual dbi_status init();
            virtual DBIHandle *connect( const String &parameters, bool persistent,
                    dbi_status &retval, String &errorMessage );
            virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
    };

}

extern Falcon::DBIServiceOracle theOracleService;

#endif /* DBI_ORACLE_H */

/* end of oracle_mod.h */

