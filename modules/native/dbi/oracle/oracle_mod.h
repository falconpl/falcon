/*
 * FALCON - The Falcon Programming Language.
 * FILE: oracle_mod.h
 *
 * Oracle driver main module interface
 * -------------------------------------------------------------------
 * Author: Steven Oliver
 * 
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef DBI_ORACLE_H
#define DBI_ORACLE_H

#include <falcon/dbi_common.h>
#include <falcon/srv/dbi_service.h>

#include <occi.h>

namespace Falcon
{
    // FIXME Oracle doesn't have an "ORACLE" structure like this...
    class ORACLEHandle : public DBIRefCounter<ORACLE*> {
        public:
            ORACLEHandle( ORACLE* o );
            DBIRefCounter<ORACLE*>( o )
            {}

        virtual ~ORACLEHandle()
        {
            // FIXME 
        }
    };

    class DBIHandleOracle : public DBIHandle
    {
        protected:
            Connection *o_conn;
            Environment *o_env;
            ORACLEHandle *o_pConn;

        public:
            DBIHandleOracle();
            DBIHandleOracle( Connection *conn );
            virtual ~DBIHandleOracle();

            virtual void close();
            virtual void terminateEnv();
            
            virtual DBIRecordset *query( const String &sql, ItemArray* params );
            virtual DBIStatement* prepare( const String &query );
            //virtual int64 getLastInsertedId( const String& name = "" ); FIXME
            
            virtual void commit();
            virtual void rollback();
            
            virtual void selectLimited( const String& query,
                    int64 nBegin, int64 nCount, String& result );
            
            ORACLEHandle *getConn() { return o_pConn; }
            ORACLEHandle *getEnv() { return o_env; }
    };

    class DBIHandleOracle;

    class DBIStatementOracle : public DBIStatement
    {
        protected:
            Statment* o_statement;
            OracleHandle* o_pConn;
            //OracleStmtHandle *o_pStmt; FIXME
            //ODBIInBind* o_inBind;      FIXME
            bool o_bBound;
            
        public:
            DBIStatementOracle( DBIHandleOracle *dbh, Statement* stmt );
            virtual ~DBIStatementOracle();
            
            virtual DBIRecordset* execute( ItemArray* params );
            virtual void close();
            
            DBIHandleOracle* getOracle() const { return static_cast<DBIHandleOracle*>( o_dbh ); }
            Statement* o_statement() const { return o_statement; }
    };

    class DBIServiceOracle : public DBIService
    {
        public:
            DBIServiceOracle() : DBIService( "DBI_oracle" ) {}
            
            virtual void init();
            virtual void DBIHandle *connect( const String &parameters );
            virtual void CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
    };
}

extern Falcon::DBIServiceOracle theOracleService;

#endif /* DBI_ORACLE_H */

/* end of oracle_mod.h */

