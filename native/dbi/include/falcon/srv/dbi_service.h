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



namespace Falcon
{

class VMachine;
class String;
class ItemArray;

class DBIStatement;
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
    * \return a configured DBIHandle or 0 on error.
    */
   virtual DBIHandle *connect( const String &parameters )=0;

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


}

#endif

/* end of dbiservice.h */
