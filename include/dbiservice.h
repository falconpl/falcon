/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbiservice.h
 *
 * DBI service that DBI drivers inherit from and implement
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Sun, 23 Dec 2007 19:22:38 +0100
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

#ifndef DBI_SERVICE_H
#define DBI_SERVICE_H

#include <falcon/engine.h>
#include <falcon/service.h>
#include <falcon/error.h>
#include <falcon/timestamp.h>

#define DBI_MAX_COLUMN_NAME_SIZE 128

namespace Falcon
{

class CoreArray;
class VMachine;
class String;

class DBIHandle;

typedef enum {
   dbit_string,
   dbit_integer,
   dbit_integer64,
   dbit_numeric,
   dbit_date,
   dbit_time,
   dbit_datetime,
   dbit_boolean,
}
dbi_type;

typedef enum {
   /** operation was OK, no error */
   dbi_ok = 0,

   /** attempted to load a driver that was not found */
   dbi_driver_not_found = 1,

   /** not implemented by dbi driver */
   dbi_not_implemented = 2,

   /** end of file (or recordset) */
   dbi_eof = 3,

   /** currently an invalid row index while a field fetch occurred */
   dbi_row_index_invalid = 4,

   /** a non-existant column was requested */
   dbi_column_range_error = 5,

   /** operation attempted while recordset is invalid, closed? */
   dbi_invalid_recordset = 6,

   /** operation attempted while connection is invalid, closed? */
   dbi_invalid_connection = 7,

   /** nil value in a database field */
   dbi_nil_value = 8,

   /** an invalid type has been requested */
   dbi_invalid_type = 9,

   /** memory could not be allocated for connection, result, transaction, etc... */
   dbi_memory_allocation_error = 10,

   /** an error has occurred executing the query: from SQL server */
   dbi_execute_error = 11,

   /** an error has occurred processing the query: from SQL server */
   dbi_query_error = 12,

   /** no results for a query */
   dbi_no_results = 13,

   /** no error exists (did you call getLastError in non-error condition?) */
   dbi_no_error_message = 14,

   /** failed to parse a sql expansion ($1, $2, ...) correctly */
   dbi_sql_expand_error = 15,

   /** failed to expand a value due to it being an unknown type */
   dbi_sql_expand_type_error = 16,

   /** failed to connect to SQL server */
   dbi_connect_error,

   /** There was a non-string item in the "_persist" property. */
   dbi_persist_has_no_string = 17,

   /** an unknown or generic error has occurred */
   dbi_error=99
}
dbi_status;

/**
 * Abstraction of recordset class.
 *
 * The recordset class is the minimal query access interface unit towards the database.
 * It represents a single database query with results. Through this class, query data
 * can be accessed.
 */
class DBIRecordset : public UserData
{
protected:
   DBIHandle *m_dbh;

public:
   DBIRecordset( DBIHandle *dbh ) { m_dbh = dbh; }

   /** Move to the next record
    * \return dbi_ok on success, s_eof on end of file reached or other dbi_status error code
    */
   virtual dbi_status next()=0;

   /**
    * Get the current row number.
    *
    * \return row index (0 based) or -1 for invalid row
    */
   virtual int getRowIndex()=0;

   /**
    * Fetch the number of rows in the recordset or -1 if unknown
    */
   virtual int getRowCount()=0;

   /**
    * Fetch the number of columns in the recordset
    */
   virtual int getColumnCount()=0;

   /**
    * Fetch the column types
    */
   virtual dbi_status getColumnTypes( dbi_type *types )=0;

   /**
    * Fetch the row headers
    */
   virtual dbi_status getColumnNames( char *names[] )=0;

   /**
    * Get a value from the current row as a string
    */
   virtual dbi_status asString( const int columnIndex, String &value )=0;

   /**
    * Get a value from the current row as an integer
    */
   virtual dbi_status asInteger( const int columnIndex, int32 &value )=0;

   /**
    * Get a value from the current row as a 64 bit integer
    */
   virtual dbi_status asInteger64( const int columnIndex, int64 &value )=0;

   /**
    * Get a value from the current row as a numeric
    */
   virtual dbi_status asNumeric( const int columnIndex, numeric &value )=0;

   /**
    * Get a value from the current row as a date
    */
   virtual dbi_status asDate( const int columnIndex, TimeStamp &value )=0;

   /**
    * Get a value from the current row as a time
    */
   virtual dbi_status asTime( const int columnIndex, TimeStamp &value )=0;

   /**
    * Get a value from the current row as a datetime
    */
   virtual dbi_status asDateTime( const int columnIndex, TimeStamp &value )=0;

   /**
    * Get value from the current row as a boolean
    */
   virtual dbi_status asBoolean( const int columnIndex, bool &value )=0;

   /**
    * Returns last error and its description.
    *
    * Internal codes and possibly their meaning are written in the description
    * field, while dbi_status return the status generated by the last operation.
    * If the last exit code was dbi_ok, nothing is written in description.
    * \param description a string where to write last error description
    * \return the last operation status
    */
   virtual dbi_status getLastError( String &description )=0;

   /**
    * Close the recordset
    */
   virtual void close()=0;
};

/**
 * Abstraction of transaction class.
 *
 * The transaction class is the minimal operative interface unit towards the database.
 * It represents a single database transaction. Through this class, queries can be
 * performed, results can be fetched and changes can be commited.
 *
 * Some databases are mono-transaction (one transaction per connection). For those
 * databases, opening new transactions except the default one is considered an error,
 * and will raise an error at script level.
 *
 * DBITransactions may live either inside the DBIHandle class that originated them
 * (representing the "default" transaction of the DB), or can be stored into
 * wrapping CoreObjects fors cripts.
 */

class DBITransaction: public UserData
{
protected:
   DBIHandle *m_dbh;

public:
   DBITransaction( DBIHandle *dbh ) { m_dbh = dbh; }

   /**
    * Get the DBIHandle associated with this transaction.
    */
   virtual DBIHandle *getHandle() { return m_dbh; }

   /** Launches a query
    * \param query SQL query to execute
    * \param retval result status of operation
    * \return DBIRecordset or NULL on error (check retval for reason,
    *    getLastError for message)
    */
   virtual DBIRecordset *query( const String &query, dbi_status &retval )=0;

   /** Launches an INSERT/UPDATE query
    * \param query SQL query to execute
    * \param retval result status of operation
    * \return number of affected rows or -1 on error (check retval for reason,
    *    getLastError() for message)
    */
   virtual int execute( const String &query, dbi_status &retval )=0;

   /** Commits operations. */
   virtual dbi_status commit()=0;

   /** Rollback the transaction. */
   virtual dbi_status rollback()=0;

   /**
    * Close the transaction.
    * This tells the DB API that this transaction will not be used anymore.
    * \note Subclasses desctructor must take care to properly close the
    *       transaction if it's still open at object destruction.
    * \note the instances are often required to be closed through their
    *       database handlers. The subclasses should have an owner field
    *       and perform closing through the close( DBITransaction *) method
    *       of the owner.
    */
   virtual void close()=0;

   /**
    * Returns last error and its description.
    * Internal codes and possibly their meaning are written in the description
    * field, while dbi_status return the status generated by the last operation.
    * If the last exit code was dbi_ok, nothing is written in description.
    * \param description a string where to write last error description
    * \return the last operation status
    */
   virtual dbi_status getLastError( String &description )=0;
};

/**
 * Base class for handlers.
 * This class holds handle to database connections.
 *
 * Database drivers must derive this and provide specific handlers
 * towards given connections.
 *
 * Shared connection management (i.e. persistent connections, resource
 * condivisions and so on) must be handled at driver level; at user
 * level, each instance of database object must receive a different
 * handler.
 *
 * The handle is derived from UserData as it is likely to be assigned
 * to a CoreObject.
 */
class DBIHandle: public UserData
{
public:
   typedef enum {
      q_no_expansion,
      q_question_mark_expansion,
      q_dollar_sign_expansion
   }
   dbh_query_expansion;

   DBIHandle() {}
   virtual ~DBIHandle() {}

   /**
    * State what type of query expansion the driver can handle.
    *
    * \return dbh_query_expansion
    */
   virtual dbh_query_expansion getQueryExpansionCapability() { return q_no_expansion; }

   /**
    * Starts a new transaction.
    * It is legal to return the default transaction even for mono-transaction db.
    * If more than one transaction is asked for non-transactional dbs, a
    * s_single_transaction error is set in the class, and 0 is returned.
    * \return a DBITransaction instance on success,
    *         0 on error (use getLastError to determine what happened).
    */
   virtual DBITransaction *startTransaction()=0;

   /**
    * Closes a transaction.
    *
    * Once called this method, the subclasses should forget about
    * the given transaction object: it will be disposed by the caller.
    * This even if the close operation wasn't succesful.
    *
    * \param tr the transaction to be closed.
    * \return dbi_ok on success, an error on failure.
    */
   virtual dbi_status closeTransaction( DBITransaction *tr )=0;

   /**
    * Perform a SQL query returning results.
    *
    * \param sql code to execute
    * \param retval result of operation
    * \return DBIRecordset to access the record information
    */
   virtual DBIRecordset *query( const String &sql, dbi_status &retval )=0;

   /**
    * Perform a SQL query that has no results.
    *
    * \param sql code to execute
    * \param retval result of operation
    * \return number of records affected
    */
   virtual int execute( const String &sql, dbi_status &retval )=0;

   /**
    * Returns the last inserted id.
    *
    * \return value of the last inserted id or -1 on error
    */
   virtual int64 getLastInsertedId()=0;

   /**
    * Returns the last inserted id.
    *
    * \param sequenceName sequence name for those databases requiring it.
    * \return value of the last inserted id or -1 on error
    */
   virtual int64 getLastInsertedId( const String &sequenceName )=0;

   /**
    * Returns last error and its description.
    *
    * Internal codes and possibly their meaning are written in the description
    * field, while dbi_status return the status generated by the last operation.
    * If the last exit code was dbi_ok, nothing is written in description.
    * \param description a string where to write last error description
    * \return the last operation status
    */
   virtual dbi_status getLastError( String &description )=0;

   /**
    * Escape a string making it suitable for inserting into a SQL query.
    */
   virtual dbi_status escapeString( const String &value, String &escaped )=0;

   /** Disconnects this handle from the database */
   virtual dbi_status close()=0;
};

/**
 * Base class for database providers.
 *
 * Database provider services must derive from this class and subscribe
 * their module with different names.
 */
class DBIService: public Service
{
protected:
   DBIService( const String &name ):
      Service( name )
   {}

public:
   /**
    * Initialization hook.
    *
    * It gets called as soon as the service is loaded. It may be used by the
    * module to initialize global data.
    *
    * \return dbi_ok if initialization is succesfull or error.
    */
   virtual dbi_status init()=0;

   /**
    * Initialization hook
    *
    * It gets called as soon as the service is loaded. It may be used by the
    * module to initialize global data.
    *
    * The function returns a DBIHandle because some driver may want to re-use
    * already existing DBI handles if they are capable to perform concurrent
    * operations and if the connection parameters are compatible with already
    * existing connections.
    *
    * \note add doc on connection parameters
    * \param parameters the connection parameters.
    * \param persistent true if the DBIHandle may be one already served DBI handle,
    *    false if it should anyhow be created anew.
    * \param retval error code if necessary
    * \param errorMessage error description if necessary and available
    * \return a configured DBIHandle or 0 on error.
    */
   virtual DBIHandle *connect( const String &parameters, bool persistent,
                               dbi_status &retval, String &errorMessage )=0;

   /**
    * Creates an instance of database handle for Falcon scripts.
    *
    * This function creates a core object getting the DBI database handler subclass
    * managed by the providing module.
    *
    * It can't fail.
    */
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh ) = 0;
};

/**
 * Service used to load DBI modules.
 *
 * This service is meant as an utility and is mainly used internally by the DBI module.
 * It resolves the module names, checks if they are currently loaded in the VM and, if not,
 * loads and links them.
 */
class DBILoader: public Service
{
protected:
   DBILoader( const String &name ):
      Service( name )
   {}

public:
   /**
    * Loads the required provider and returns the service it provides.
    *
    * If the service is already present in the VM, that service is returned instead.
    * The VM is used as the error handler of the loader that will load the modules,
    * so, in case of errors, the VM will be notified with a standard module loading
    * error.
    *
    * \return a DBIService instance or 0
    */
   virtual DBIService *loadDbProvider( VMachine *vm, const String &provName )=0;

};

/**
 * Error for all DBI errors.
 */
class DBIError: public ::Falcon::Error
{
public:
   DBIError():
      Error( "DBIError" )
   {}

   DBIError( const ErrorParam &params  ):
      Error( "DBIError", params )
      {}
};

}

#endif

/* end of dbiservice.h */
