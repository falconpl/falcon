/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_trans.h

   Database Interface - SQL Transaction class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 May 2010 00:09:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_DBI_TRANS_H_
#define FALCON_DBI_TRANS_H_

#include <falcon/falcondata.h>
#include <falcon/string.h>

namespace Falcon
{
class DBIHandle;
class DBISettingParams;
class DBIRecordset;
class ItemArray;


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

   virtual void gcMark( uint32 );
   virtual FalconData* clone() const;

protected:
   DBIHandle *m_dbh;
   DBISettingParams* m_settings;
};

}

#endif

/* end of dbi_trans.h */
