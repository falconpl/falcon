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

#include <falcon/falcondata.h>
#include <falcon/string.h>

namespace Falcon
{

class DBIStatement;
class DBIRecordset;
class DBISettingParams;
class ItemArray;

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

   DBIHandle();
   virtual ~DBIHandle();

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
   virtual void perform( const String &sql, int64 &affectedRows, const ItemArray& params )=0;

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

   virtual void gcMark( uint32 );
   virtual FalconData* clone() const;
};

}

#endif

/* end of dbi_handle.h */
