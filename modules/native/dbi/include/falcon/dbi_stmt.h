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


/** Abstraction of statement class.


*/

class DBIStatement: public FalconData
{

public:
   DBIStatement( DBIHandle *dbh );
   virtual ~DBIStatement();

   /** prepare/execute step2
    * @param params the parameters.
    *
   */
   virtual DBIRecordset* execute( ItemArray* params = 0 ) =0;

   virtual void reset()=0;

   /** Closes the transaction.
    *
    *  By default, the engine must also close the transaction.
    */
   virtual void close()=0;

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

   /** Get the DBIHandle associated with this transaction.
       */
   DBIHandle *getHandle() const { return m_dbh; }

   virtual void gcMark( uint32 );
   virtual FalconData* clone() const;
   
   /** returns the count of rows affected by the last query() operation */
   int64 affectedRows();

protected:
   DBIHandle *m_dbh;
   int64 m_nLastAffected;
};

}

#endif

/* end of dbi_trans.h */
