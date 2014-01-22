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

#define FALCON_DBI_HANDLE_SERVICE_NAME "DBIService"

#include <falcon/engine.h>
#include <falcon/service.h>
#include <falcon/error.h>
#include <falcon/timestamp.h>

#include <falcon/error_base.h>

namespace Falcon {

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

/**
 * Base class for database providers.
 *
 * Database provider services must derive from this class and subscribe
 * their module with different names.
 */
class DBIService: public Service
{
protected:
   DBIService( const String &name, Module* master ):
      Service( name, master )
   {}

public:
   virtual ~DBIService() {}

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

   /** Itemization hook.
    *
    * Actually does nothing.
    */
   virtual void itemize( Item& ) const {};
};

}

#endif

/* end of dbiservice.h */
