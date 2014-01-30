/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_handle.h

   Database Interface - Main handle driver
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 May 2010 00:09:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_DBI_HANDLE_H_
#define FALCON_DBI_HANDLE_H_

#define FALCON_DBI_HANDLE_CLASS_NAME "%Handle"

#include <falcon/string.h>

namespace Falcon {
class ItemArray;

class DBIStatement;
class DBIRecordset;
class DBISettingParams;

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
 */
class DBIHandle
{
public:

   DBIHandle( const Class* h );
   virtual ~DBIHandle();

   /** Creates a connection with the database system.
    *
    */
   virtual void connect( const String& params ) = 0;

   /** Sets the common transaction options.
    *
    *    Used to change the default values for transaction creation.
    *
    * @param params Parameters for the transaction (see DBISettingParams).
    * @return true on success, false on parse error
    */
   virtual void options( const String& params ) = 0;

   /** Return the transaction settings used as the default options by this connection. */
   virtual const DBISettingParams* options() const = 0;

   virtual void begin() = 0;
   virtual void commit() = 0;
   virtual void rollback() = 0;

   /** Writes a select query with limited bounds that is valid for the engine.

       This method should create a "select" query adding the commands and/or the
       parameters needed by the engine to limit the resultset to a specified part
       part of the dataset.

       The query parameter must be a complete query EXCEPT for the "select" command,
       which is added by the engine. It must NOT terminate with a ";", which, in case
       of need is added by the engine.

       For example, the following query
       @code
          SELECT field1, field2 FROM mytable WHERE key = 2;
       @endcode

       should be passed as
       @code
          field1, field2 FROM mytable WHERE key = 2
       @endcode

       An engine must at least add the "select" command and return the modified
       query i8n the result output parameter. If supported, it must modify
       the query so that it contains informations to skip the records selected
       up to nBegin (0 based), and to return nCount rows.

       The nCount parameter will be 0 to indicate "from nBegin to the end".
       It's not possible to return the n-last elements; to do that, reverse the
       query ordering logic.

     @param query The SQL statement stripped of the initial "select" and of the final ";"
     @param nBegin First row to be returned (0-based).
     @param nCount Number of rows to be returned in the recordset.
     @param result The SQL query statement correctly modified for the engine to parse it.
   */
   virtual void selectLimited( const String& query,
         int64 nBegin, int64 nCount, String& result ) = 0;


   /** Launches a query (an SQL operation bound to return a recordset).
    *
    * \param sql SQL query to execute
    * \param params An array of items that will be used to expand query variables.
    * \return DBIRecordset if there is an output recordset.
    *     NULL if the query has an error.
    */
   virtual DBIRecordset *query( const String &sql, ItemArray* params=0 )=0;

   /** Prepare/execute step1
    */
   virtual DBIStatement* prepare( const String &query )=0;

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
    * Close the connection with the Database.
    * This tells the DB API that this database will not be used anymore.
    */
   virtual void close()=0;

   /**
    * Utility performing direct sql expansion.
    *
    * This utility transforms question marks into values (properly formatted and escaped)
    * to be used in SQL statements.
    *
    * Will throw an adequate DBI error in case of expansion error.
    */
   virtual void sqlExpand( const String& sql, String& tgt, const ItemArray& values );
   virtual void gcMark( uint32 mark ) { m_mark = mark; }
   uint32 currentMark() const { return m_mark; }

   /** returns the count of rows affected by the last query() operation */
   int64 affectedRows();

   const Class* handler() const { return m_handler; }


protected:
   int64 m_nLastAffected;
   uint32 m_mark;
   const Class* m_handler;
};

}

#endif

/* end of dbi_handle.h */
