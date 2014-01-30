/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_drivermod.h

   Base driver main module

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 30 Jan 2014 12:58:50 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DBI_DRIVERMOD_H_
#define _FALCON_DBI_DRIVERMOD_H_

#include <falcon/module.h>
#include <falcon/dbi_service.h>

namespace Falcon {

/**
 * Base driver module.
 *
 * This class forms the base of a DBI driver module, providing
 * common behaviro for all DBI drivers:
 * # load the DBI main module
 * # get the DBIHandle class from that module
 * # add the DBIHandle class to the parentship of the DBIHandle subclass
 *   created by that module
 * # publish the connect service to DBI main module.
 */

class DriverDBIModule: public Module
{
public:
   DriverDBIModule( const String& name );
   virtual ~DriverDBIModule();

   virtual void onImportResolved( ImportDef* id, Symbol* sym, Item* value );
   virtual void onLinkComplete( VMContext* ctx );
   Class* driverDBIHandle() const { return m_driverDBIHandle; }

   virtual Service* createService( const String& name );

protected:
   /**
    * Driver class created by the final module.
    *
    * This is the DBIHandle sub-class that is created by the submodule.
    * The constructor has to create it and set this variable to that value,
    * and then add the class to the mantras of the module.
    *
    * This subclass will use the given value to fill its parentship.
    */
   Class* m_driverDBIHandle;

private:
   Class* m_dbiHandle;

   class DriverService: public DBIService {
   public:
      DriverService( Module* master );
      virtual ~DriverService();
      virtual DBIHandle *connect( const String &parameters );
   };
   Service* m_dbiService;
};

}

#endif

/* end of dbi_drivermod.h */
