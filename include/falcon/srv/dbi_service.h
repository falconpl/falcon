/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_service.h
 *
 * DBI service that DBI drivers inherit from and implement
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Sun, 23 Dec 2007 19:22:38 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007-2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef DBI_SERVICE_H
#define DBI_SERVICE_H

#include <falcon/engine.h>
#include <falcon/service.h>
#include <falcon/error.h>
#include <falcon/timestamp.h>

#include <falcon/error_base.h>

#ifndef FALCON_DBI_ERROR_BASE
   #define FALCON_DBI_ERROR_BASE 2000
#endif

// IMPORTANT: keep in parallel with dbi_st
#define FALCON_DBI_ERROR_COLUMN_RANGE     (FALCON_DBI_ERROR_BASE+1)
#define FALCON_DBI_ERROR_INVALID_DRIVER   (FALCON_DBI_ERROR_BASE+2)
#define FALCON_DBI_ERROR_NOMEM            (FALCON_DBI_ERROR_BASE+3)
#define FALCON_DBI_ERROR_CONNPARAMS       (FALCON_DBI_ERROR_BASE+4)
#define FALCON_DBI_ERROR_CONNECT          (FALCON_DBI_ERROR_BASE+5)
#define FALCON_DBI_ERROR_QUERY            (FALCON_DBI_ERROR_BASE+6)
#define FALCON_DBI_ERROR_QUERY_EMPTY      (FALCON_DBI_ERROR_BASE+7)
#define FALCON_DBI_ERROR_OPTPARAMS        (FALCON_DBI_ERROR_BASE+8)
#define FALCON_DBI_ERROR_NO_SUBTRANS      (FALCON_DBI_ERROR_BASE+9)
#define FALCON_DBI_ERROR_NO_MULTITRANS    (FALCON_DBI_ERROR_BASE+10)
#define FALCON_DBI_ERROR_UNPREP_EXEC      (FALCON_DBI_ERROR_BASE+11)
#define FALCON_DBI_ERROR_BIND_MIX         (FALCON_DBI_ERROR_BASE+12)

namespace Falcon
{

class VMachine;
class String;
class ItemArray;

class DBITransaction;
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
   dbit_blob
}
dbi_type;



/** An utility used by many drivers to break connection strings.
   Base class for more specific
*/
class DBIParams
{

protected:
   /** A Specification of a connection parameter.

      The parameter is a pair of a parameter name (as "uid" or "pwd") and
      of the output parameter, that is a pointer to a string that will be filled
      with the parameter name.
   */
   class Param
   {
   public:
      Param( const String& name, String& output ):
         m_name( name ),
         m_output( output ),
         m_szOutput( 0 ),
         m_cstrOut(0),
         m_pNext(0)
         {}

      Param( const String& name, String& output, const char** szOutput ):
         m_name( name ),
         m_output( output ),
         m_szOutput( szOutput ),
         m_cstrOut(0),
         m_pNext(0)
         {}

      ~Param();

      /** Parses an input string.
       * @param value A string in format <name>=<value> or <name>=
       * @return True if the name matches (case insensitive) the name of this parameter.
       */
      bool parse( const String& value );

      String m_name;
      String &m_output;
      const char** m_szOutput;
      AutoCString* m_cstrOut;
      Param* m_pNext;
   };

   Param* m_pFirst;

   bool parsePart( const String& strPart );

   DBIParams();

public:
   virtual ~DBIParams();

   /** Function adding a parse parameter */
   virtual void addParameter( const String& name, String& value );

   /** Function adding a parse parameter and its c-string value */
   virtual void addParameter( const String& name, String& value, const char** szValue );

   /** Parse the parameter=value string */
   virtual bool parse( const String& connStr );
};


/** Parameter parser for the settings string.

    The setting string is used at DBIConnect,
    Connection.setOptions and Connection.tropen;
    they determine the behavior of db fetches, and may
    be implemented differently by different database drivers.

    Setting options are:

    - prefetch: number of records to be pre-fetched at each query. The value
                may be "all" to wholly fetch queries locally, "none" to prefetch
                none or an arbitrary number of rows to be read from the server.
                By default, it's "all".
    - autocommit: Performs a transaction commit after each sql command.
                  Can be "on" or "off"; it's "off" by default.
    - cursor: Number of records returned by a query that should trigger the creation of a
              server side cursor. Can be "none" to prevent creation of server
              side cursor (the default) "all" to always create a cursor or an arbitrary
              number to create a server side cursor only if the query returns at least
              the indicated number of rows.
    - name: Some engine allow to create named transactions that can be directly referenced
            in SQL Statements from other transactions. If this feature is supported by the
            engine, the transaction receives this name, otherwise the option is ignored.

    After a complete local prefetch, all the records
    are moved to the client, so it's possible to issue another query returning a different
    recordset. If moving all the returned rows to the client is not feasible, but it's still
    necessary to fetch rows from a query and performing other queries based on the retrieved
    results, then it's necessary to create a cursor on the server side. Many SQL engines allow
    to specify to open the cursor directly on the SQL statement, but this option is given
    to provide the user with the ability to access this feature without using engine-specific
    SQL.

    Notice that server-side cursor and complete local prefetch are alternative methods
    to make the recordset consistent across different queries in the same transaction,
    so it's pretty useless to use both of them at the same time.

    This class transforms the given options in C variables that can easily be read by
    the drivers. Also, drivers can add their own option parsing code by overriding the
    DBIConnection::parseOptions method.
*/

class DBISettingParams: public DBIParams
{
private:
   String m_sCursor;
   String m_sAutocommit;
   String m_sPrefetch;

   static const bool defaultAutocommit = false;
   static const int defaultCursor = -1;
   static const int defaultPrefetch = -1;

public:
   DBISettingParams();
   DBISettingParams( const DBISettingParams & other );
   virtual ~DBISettingParams();

   /** Specific parse analizying the options */
   virtual bool parse( const String& connStr );

   /** True if the transaction should be autocommit, false otherwise. */
   bool m_bAutocommit;

   /** Cursor invocation treshold.
    * Will be -1 if cursor should be never used, 0 if always used,
    * a positive number for a given treshold.
    */
   int64 m_nCursorThreshold;

   /** Number of rows to be pre-fetched after queries.
      Will be -1 if recordset must be completely prefetched, 0 to disable pre-fetching
      and a positive number to ask for prefetching of that many rows.
   */
   int64 m_nPrefetch;

   /** Name for this transaction. */
   String m_sName;

};


/** An utility used by many drivers to break connection strings.

   This is just an utility class that a driver may use to break connection
   strings into parameters for its DB system. A driver is free not to use
   it if, for example, it thinks it should pass the string as-is to the
   underlying DB system.

   The format of the string respects ODBC standards ("parameter=value;..."),
   so the connection stings for systems that don't rely on that to initialize the connections
   (as, for example SQLlite and MySQL) looks like ODBC connection, giving
   a flavor of portability of the connection string across different system.

   The utility works like this: a set of parameters are declared via the
   method addParameter(); other than the parameter name, this method allows
   to declare a parameter value, that is, a locally defined string where
   the parameter value will be placed if found, and optionally a C zero terminated
   string where to put the AutoCString result (converting the string in utf-8),
   in case the driver needs this  feature.

   If the required parameter is not found in the parsed string, the string is left
   empty, and the C ZTstring, if provided, is set to a null pointer. If it's found,
   but empty, then the string is set to a pair of double quotes (""), and the c ZTString,
   if provided, is set to an empty string.

   To ensure a minimal coherence across different drivers, and to reduce the effort of the
   implementers, a minimal set of common parameters are added by the constructor of this
   class. The variables where the parameter values are placed are provided by the class
   as well. Namely:

   - uid= The user id for the connection (values placed in m_sUser and m_szUser).
   - pwd= Password for the connection (values placed in m_sPassword and m_szPassword).
   - db= Name of the DB to open (values in m_sDb and m_szDb).
   - host= Host where to perform the connection (values placed in m_sHost and m_szHost).
   - port= TCP Port where the server is listening (values in m_sPort and m_szPort).
*/
class DBIConnParams: public DBIParams
{
public:
   DBIConnParams( bool bNoDefaults = false );
   virtual ~DBIConnParams();

   // Base parameters known by all systems.
   String m_sUser;
   String m_sPassword;
   String m_sHost;
   String m_sPort;
   String m_sDb;

   const char* m_szUser;
   const char* m_szPassword;
   const char* m_szHost;
   const char* m_szPort;
   const char* m_szDb;
};



#if 0
/**
 * Abstraction of blob stream class.
 *
 * The DBIBlobStream class holds a stream specialized in blob writing.
 * It just provides an extra BlobID entry and driver specific setting
 * parser that will be hooked by the drivers.
 */
class DBIBlobStream : public Stream
{
   String m_blobID;
   String m_falconClassName;

protected:
   DBIBlobStream():
      Stream( t_membuf ),
      m_falconClassName( "DBIBlobStream")
   {}

   /**
      Sets a falcon specific class name that should wrap this subclass.
   */
   DBIBlobStream( const String &className ):
      Stream( t_membuf ),
      m_falconClassName( className )
   {}

public:

   void setBlobID( const String &blobID ) { m_blobID = blobID; }
   const String &getBlobID() { return m_blobID; }

   /** Retiurns the Falcon class name that encapsulates this stream.
      If drivers don't want to bother creating a new Stream class for
      their own blobs, they can just use the default DBIBlobStream class
      declared in the base DBI.

      However, submodules may be willing to provide more functionalities
      on open blobs, in example, they may return informations about the
      blob size, fragments, type and so on. In that case, specific per-module
      blob class name may be returned; In case an encapsulation is needed
      by the script, the DBI module will ask the VM to instance the required
      blob class, and will use that to the Falcon::DBIBlobStream instance.
   */
   virtual const String &getFalconClassName() const { return m_falconClassName; }
};

#endif

/**
 * Abstraction of recordset class.
 *
 * The recordset class is the minimal query access interface unit towards the database.
 * It represents a single database query with results. Through this class, query data
 * can be accessed.
 */
class DBIRecordset : public FalconData
{

public:
   DBIRecordset( DBITransaction* generator ):
      m_trh( generator )
   {}

   virtual ~DBIRecordset()
   {}

   /** Move to the next record
    * \throw DBIError* in case of error.
    * \return true on success, false on end of updates reached
    */
   virtual bool fetchRow()=0;

   /**
    * Get the current row number.
    *
    * \return row index (0 based) or -1 for invalid row
    */
   virtual int64 getRowIndex()=0;

   /**
    * Fetch the number of rows in the recordset or -1 if unknown
    */
   virtual int64 getRowCount()=0;

   /**
    * Fetch the number of columns in the recordset
    */
   virtual int getColumnCount()=0;

   /**
    * Fetch the row headers
    */
   virtual bool getColumnName( int nCol, String& name )=0;

   /** Gets a value in the recordset.
    */
   virtual bool getColumnValue( int nCol, Item& value )=0;

   /** Gets a type in the recordset.
    */
   //virtual dbi_status getColumnType( int nCol, dbi_type& type )=0;

   /** Skip the required amount of records from this position on. */
   virtual bool discard( int64 ncount ) = 0;

   /**
    * Close the recordset
    */
   virtual void close()=0;

   //=========================================================
   // Manage base class control.
   //
   virtual FalconData *clone() const { return 0; }
   virtual void gcMark( uint32 ) {}

protected:
   DBITransaction* m_trh;
};



/**
 * Abstraction of transaction class.
 *
 * The transaction class is the minimal operative interface unit towards the database.
 * It represents a single database transaction. Through this class, queries can be
 * performed, results can be fetched and changes can be committed.
 *
 * Some databases are mono-transaction (one transaction per connection). For those
 * databases, opening new transactions except the default one is considered an error,
 * and will raise an error at script level.
 *
 * DBITransactions may live either inside the DBIHandle class that originated them
 * (representing the "default" transaction of the DB), or can be stored into
 * wrapping CoreObjects for scripts.
 *
 * Transaction subclasses must also keep tracks of the parameter they receive.
 * Parameters are set via an instance of the DBISettingParams, which are owned
 * and deleted by the DBITransaction instance on exit.
 *
 * Subclasses may use different instances of DBISettingsParams. However, it shouldn't
 * be allowed to change the parameters after the transaction creation.
 */

class DBITransaction: public FalconData
{

public:
   DBITransaction( DBIHandle *dbh, DBISettingParams* params );
   virtual ~DBITransaction();

   /** Launches a query (an SQL operation bound to return a recordset).
    *
    * \param sql SQL query to execute
    * \param affectedRows number of rows affected by the query.
    * \param params An array of items that will be used to expand query variables.
    * \return DBIRecordset if there is an output recordset.
    *     NULL if the query has an error.
    */
   virtual DBIRecordset *query( const String &sql, int64 &affectedRows, const ItemArray& params )=0;

   /** Launches a SQL operation not bound to return any recordset.
    *
    * If the statement actually returns a recordset, it is discarded.
    *
    * \param sql SQL statement to execute.
    * \param affectedRows number of columns affected by the statement.
    * \param params An array of items that will be used to expand query variables.
    */
   virtual void call( const String &sql, int64 &affectedRows, const ItemArray& params )=0;

   /** Prepare/execute step1
    */
   virtual void prepare( const String &query )=0;

   /** prepare/execute step2
   */
   virtual void execute( const ItemArray& params )=0;

   /** Commits operations. */
   virtual void commit()=0;

   /** Rollback the transaction. */
   virtual void rollback()=0;

   /** Closes the transaction.
    *
    *  By default, the engine must also close the transaction.
    *
    *  After the transaction is closed, the subclasses must make sure
    *  that the
    */
   virtual void close()=0;


   /** Returns the last inserted ID.
    *
    *  Many engines provide this feature so that the last inserted ID auto-generated
    *  number in the last inserted translation can be retrieved.
    *
    *  Return -1 if the engine doesn't provide this feature; 0 is the common return
    *  value when no auto-increment ID has been inserted.
    */
   virtual int64 getLastInsertedId( const String& name = "" )=0;

   /**
    * Open a blob entity.
    * \param blobId the ID through which the field is known.
    * \param stauts a DBI error code (on error).
    * \return On success, an open stream to the blob, or 0 on failure.
    * The BlobID database-specific, and may also be a fully binary value (i.e. a 64 bit numeric ID).
    * It's responsibility of the driver to correctly decode the contents of the blob.
    */
   //virtual DBIBlobStream *openBlob( const String &blobId, dbi_status &status )=0;

   /**
    * Create a new blob entity.
    * \param stauts a DBI error code (on error).
    * \param params Driver specific settings (i.e. blob subtype).
    * \param bBinary The caller will set this to true to override generic parameters.
    * \return On success, an open stream to the blob, or 0 on failure.
    *
    * The BlobID database-specific, and may also be a fully binary value (i.e. a 64 bit numeric ID).
    * It's responsibility of the driver to correctly decode the contents of the blob.
    *
    * If bBinary is true, the drivers should try to create a binary-oriented blob, while
    * if its false, they should use a text-oriented blob. Driver specific parameters must
    * override this setting, which must be ignored in case a specific setting is provided.
    */
   //virtual DBIBlobStream *createBlob( dbi_status &status, const String &params= "", bool bBinary = false )=0;

   /** Starts a sub-transaction */
   virtual DBITransaction* startTransaction( const String& options ) =0;

   /** Get the DBIHandle associated with this transaction.
       */
   DBIHandle *getHandle() const { return m_dbh; }

   virtual const DBISettingParams* params() const { return m_settings; }

   virtual void gcMark( uint32 ) {};
   virtual FalconData* clone() const { return 0; }

protected:
   DBIHandle *m_dbh;
   DBISettingParams* m_settings;
};


/**
 * Base class for handlers.
 * This class holds handle to database connections.
 *
 * Database drivers must derive this and provide specific handlers
 * towards given connections.
 *
 * Shared connection management (i.e. persistent connections, resource
 * sharing and so on) must be handled at driver level; at user
 * level, each instance of database object must receive a different
 * handler.
 *
 * The handle is derived from UserData as it is likely to be assigned
 * to a CoreObject.
 */
class DBIHandle: public FalconData
{
public:

   DBIHandle() {}
   virtual ~DBIHandle() {}

   /** Sets the common transaction options.
    *
    *    Used to change the default values for transaction creation.
    *
    * @param params Parameters for the transaction (see DBISettingParams).
    * @return true on success, false on parse error
    */
   virtual bool setTransOpt( const String& params ) = 0;

   /** Return the transaction settings used as the default options by this connection. */
   virtual const DBISettingParams* transOpt() const = 0;

   /**
    * Starts a new transaction.
    *
    * If more than one transaction is asked for non-transactional dbs, a
    * s_single_transaction error is set in the class, and 0 is returned.
    * \return a DBITransaction instance on success,
    *         0 on error (use getLastError to determine what happened).
    */
   virtual DBITransaction *startTransaction( const String& options )=0;

    /**
    * Close the connection with the Database.
    * This tells the DB API that this database will not be used anymore.
    */
   virtual void close()=0;

   virtual void gcMark( uint32 ) {}
   virtual FalconData* clone() const { return 0; }

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
   virtual ~DBIService() {}

   /**
    * Initialization hook.
    *
    * It gets called as soon as the service is loaded. It may be used by the
    * module to initialize global data.
    */
   virtual void init()=0;


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
    * \return a configured DBIHandle or 0 on error.
    */
   virtual DBIHandle *connect( const String &parameters, bool persistent )=0;

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

  //virtual void escapeString( const String &value, String &escaped ) = 0;
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
