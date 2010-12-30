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
    class ORACLEHandle;


    class ORACLEStmtHandle: public DBIRefCounter<oracle::occi::Statement*> {
        public:
            ORACLEStmtHandle( oracle::occi::Statement* o ):
                DBIRefCounter<oracle::occi::Statement*>( o )
        {}
            
            virtual ~ORACLEStmtHandle()
            {
                //mysql_stmt_close( handle() ); FIXME
            }
    };


    class ODBIInBind: public DBIInBind
    {
        
        public:
            ODBIInBind( oracle::occi::Statement * stmt );
            
            virtual ~ODBIInBind();
            
            virtual void onFirstBinding( int size );
            virtual void onItemChanged( int num );
            
            //MYSQL_BIND* mybindings() const { return m_mybind; } FIXME
            
        private:
            //MYSQL_BIND* m_mybind;
            oracle::occi::Statement* o_stmt;
    };


    class ODBIOutBind: public DBIOutBind
    {
        public:
            ODBIOutBind():
                //bIsNull( false ),
                nLength( 0 ) 
        {}
            
            ~ODBIOutBind() {}
            
            //o_bool bIsNull;
            unsigned long nLength;
    };


    class DBIHandleOracle;


    class DBIRecordsetOracle : public DBIRecordset
    {
        protected:
            int o_row;
            int o_rowCount;
            int o_columnCount;
            
            bool m_bCanSeek;
            
            ORACLEHandle *o_pConn;
            
        public:
            virtual ~DBIRecordsetOracle();
            
            virtual int64 getRowIndex();
            virtual int64 getRowCount();
            virtual int getColumnCount();
            virtual bool getColumnName( int nCol, String& name );
            virtual void close();
    };

    
    class DBIHandleOracle : public DBIHandle
    {
        protected:
            oracle::occi::Connection *o_conn;
            oracle::occi::Environment *o_env;
            DBISettingParams o_settings;
            
        public:
            DBIHandleOracle();
            DBIHandleOracle( oracle::occi::Connection *conn );
            virtual ~DBIHandleOracle();
            
            //virtual void options( const String& params );     FIXME
            //virtual const DBISettingParams* options() const;
            virtual void close();
            
            //virtual DBIRecordset *query( const String &sql, ItemArray* params ); FIXME
            //virtual DBIStatement* prepare( const String &query );
            
            virtual void commit();
            virtual void rollback();
            
            //virtual void selectLimited( const String& query, int64 nBegin, int64 nCount, String& result ); FIXME
    };


    class DBIStatementOracle : public DBIStatement
    {
        protected:
            oracle::occi::Statement* o_statement;
            ODBIInBind* o_inBind;      
            bool o_bBound;
            
        public:
            DBIStatementOracle( DBIHandleOracle *dbh, oracle::occi::Statement* stmt );
            virtual ~DBIStatementOracle();
            
            virtual DBIRecordset* execute( ItemArray* params );
            virtual void close();
            
            oracle::occi::Statement* my_statement() const { return o_statement; }
    };


    class DBIServiceOracle : public DBIService
    {
        public:
            DBIServiceOracle() : DBIService( "DBI_oracle" ) {}
            
            virtual void init();
            virtual DBIHandle *connect( const String &parameters );
            virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
    };
}

extern Falcon::DBIServiceOracle theOracleService;

#endif /* DBI_ORACLE_H */

/* end of oracle_mod.h */

