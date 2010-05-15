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

namespace Falcon
{

class String;
class DBITransaction;
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

   virtual void gcMark( uint32 );
   virtual FalconData* clone() const;
};

}

#endif

/* end of dbi_handle.h */
