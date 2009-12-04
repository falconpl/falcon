/*
 * FALCON - The Falcon Programming Language.
 * FILE: oracle_mod.h
 *
 * Oracle driver main module interface
 * -------------------------------------------------------------------
 * Author: Steven Oliver
 * Based on equivlent MySQL drivers by Jeremy Cowgar
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

            Connection *getConn() { return m_conn; }

            virtual DBITransaction *startTransaction();
            virtual dbi_status closeTransaction( DBITransaction *tr );
            virtual int64 getLastInsertedId();
            virtual int64 getLastInsertedId( const String &value );
            virtual dbi_status getLastError( String &description );
            virtual dbi_status escapeString( const String &value, String &escaped );
            virtual dbi_status close();
            virtual DBITransaction* getDefaultTransaction();
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

