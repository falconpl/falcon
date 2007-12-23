/*
  FALCON - The Falcon Programming Language
  FILE: pgsql_mod.h
  
  PgSQL DBI module -- module service classes
  -------------------------------------------------------------------
  Author: Jeremy Cowgar
  Begin: 2007-12-22 10:06
  Last modified because:
  
  -------------------------------------------------------------------
  (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
  
  See LICENSE file for licensing details.
  In order to use this file in its compiled form, this source or
  part of it you have to read, understand and accept the conditions
  that are stated in the LICENSE file that comes boundled with this
  package.
*/

/** \file
	 pgsql_mod.h - PgSQL DBI module -- module service classes
*/

#ifndef flc_pgsql_mod_H
#define flc_pgsql_mod_H

#include <libpq-fe.h>

#include <dbi/dbi_mod.h>
#include <dbi/errorcodes.h>
#include <falcon/string.h>

namespace Falcon
{
	
   class PgSQLConnection;
   class PgSQLRecordset;

   class PgSQLConnection : public DBIConnection
   {
   private:
      PGconn *m_conn;

   public:
      PgSQLConnection( const String *connString ) {
         m_conn = NULL;
      }

      virtual int connect( const String *connString );

      virtual int beginTransaction();
      virtual int rollbackTransaction();
      virtual int commitTransaction();

      virtual int execute( const String *sql );
      virtual DBIRecordset *query( const String *sql );

      virtual int close();
   };

   class PgSQLRecordset : public DBIRecordset
   {
   private:
      PGresult *m_res;

   public:
      PgSQLRecordset( PgSQLConnection *connClass, PGresult *res );

      virtual int columnIndex( const String *columnName );
      virtual int columnName( const int columnIndex, String &name );
      virtual int value( const int columnIndex, String &value );

      virtual int next();
      virtual int close();
   };

}

#endif

/* end of pgsql_mod.h */
