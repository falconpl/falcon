/*
   FALCON - The Falcon Programming Language.
   FILE: sqlite3_fm.cpp

   SQLite3 driver main module

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 May 2010 18:25:58 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/dbi/sqlite3/sqlite3_fm.cpp"

#include "sqlite3_mod.h"
#include "sqlite3_ext.h"
#include "version.h"

#include "sqlite3_fm.h"

#include <falcon/importdef.h>
#include <falcon/log.h>
#include <falcon/stderrors.h>
#include <falcon/symbol.h>
#include <falcon/dbi_handle.h>

namespace Falcon {

Sqlite3DBIModule::Sqlite3DBIModule():
         Module("dbi.sqlite3")
{
   m_dbiHandle = 0;
   m_dbiService = new Sqlite3Service(this);
   Error* def = 0;
   addImport( new ImportDef("dbi", false, FALCON_DBI_HANDLE_CLASS_NAME, FALCON_DBI_HANDLE_CLASS_NAME, false), def, 0 );

   m_classSql3liteDBIHandle = new Ext::ClassSqlite3DBIHandle;
   *this
      << m_classSql3liteDBIHandle;
}


Sqlite3DBIModule::~Sqlite3DBIModule()
{
   delete m_dbiHandle;
}


void Sqlite3DBIModule::onImportResolved( ImportDef*, Symbol* sym, Item* value )
{
   if( sym->name() == FALCON_DBI_HANDLE_CLASS_NAME )
   {
      Engine::instance()->log()->log(Log::fac_engine_io, Log::lvl_detail, "Linking DBIHandle class from DBI module");
      m_dbiHandle = static_cast<Class*>(value->asInst());
      m_classSql3liteDBIHandle->setParent( m_dbiHandle );
   }
}

void Sqlite3DBIModule::onLinkComplete( VMContext* )
{
   // this is an overkill, the link system should have raised already.
   if( m_dbiHandle == 0 )
   {
      throw new LinkError(ErrorParam(e_mod_notfound, __LINE__, SRC).extra("dbi"));
   }
}

Sqlite3DBIModule::Sqlite3Service::Sqlite3Service( Module* master ):
         DBIService( FALCON_DBI_HANDLE_SERVICE_NAME, master )
{}

Sqlite3DBIModule::Sqlite3Service::~Sqlite3Service()
{}

DBIHandle *Sqlite3DBIModule::Sqlite3Service::connect( const String &parameters )
{
   Sqlite3DBIModule* mod = static_cast<Sqlite3DBIModule*>(module());
   DBIHandleSQLite3* handle = new DBIHandleSQLite3(mod->classSql3liteDBIHandle());
   handle->connect(parameters);
   return handle;
}


Service* Sqlite3DBIModule::createService( const String& name )
{
   if( name == FALCON_DBI_HANDLE_SERVICE_NAME )
   {
      return m_dbiService;
   }
   return 0;
}

}

/*#
   @module dbi.sqlite3 Sqlite driver module
   @brief DBI extension supporting sqlite3 embedded database engine

   Directly importable as @b dbi.sqlite3, it is usually loaded through
   the @a dbi module.
*/

FALCON_MODULE_DECL
{
   return new Falcon::Sqlite3DBIModule;
}

/* end of sqlite3_fm.cpp */
