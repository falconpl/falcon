/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_drivermod.cpp

   Base driver main module

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 30 Jan 2014 12:58:50 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/dbi/dbi_common/dbi_drivermod.cpp"

#include <falcon/importdef.h>
#include <falcon/log.h>
#include <falcon/stderrors.h>
#include <falcon/symbol.h>
#include <falcon/dbi_handle.h>
#include <falcon/dbi_drivermod.h>

namespace Falcon {

DriverDBIModule::DriverDBIModule( const String& name ):
         Module(name)
{
   // the subclass has to fix it
   m_driverDBIHandle = 0;

   m_dbiHandle = 0;
   m_dbiService = new DriverService(this);
   Error* def = 0;
   addImport( new ImportDef("dbi", false, FALCON_DBI_HANDLE_CLASS_NAME, FALCON_DBI_HANDLE_CLASS_NAME, false), def, 0 );
}


DriverDBIModule::~DriverDBIModule()
{
}


void DriverDBIModule::onImportResolved( ImportDef*, Symbol* sym, Item* value )
{
   if( sym->name() == FALCON_DBI_HANDLE_CLASS_NAME )
   {
      Engine::instance()->log()->log(Log::fac_engine_io, Log::lvl_detail, "Linking DBIHandle class from DBI module");
      m_dbiHandle = static_cast<Class*>(value->asInst());
      m_driverDBIHandle->setParent( m_dbiHandle );
   }
}

void DriverDBIModule::onLinkComplete( VMContext* )
{
   // this is an overkill, the link system should have raised already.
   if( m_dbiHandle == 0 )
   {
      throw new LinkError(ErrorParam(e_mod_notfound, __LINE__, SRC).extra("dbi"));
   }
}

DriverDBIModule::DriverService::DriverService( Module* master ):
         DBIService( FALCON_DBI_HANDLE_SERVICE_NAME, master )
{}

DriverDBIModule::DriverService::~DriverService()
{}

DBIHandle *DriverDBIModule::DriverService::connect( const String &parameters )
{
   DriverDBIModule* mod = static_cast<DriverDBIModule*>(module());

   DBIHandle* handle = static_cast<DBIHandle*>(mod->driverDBIHandle()->createInstance());
   handle->connect(parameters);
   return handle;
}


Service* DriverDBIModule::createService( const String& name )
{
   if( name == FALCON_DBI_HANDLE_SERVICE_NAME )
   {
      return m_dbiService;
   }
   return 0;
}

}

/* end of dbi_drivermod.cpp */
