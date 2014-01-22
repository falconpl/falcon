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

#ifndef _FALCON_DBI_SQLITE3_H_
#define _FALCON_DBI_SQLITE3_H_

#include <falcon/module.h>

namespace Falcon {

class Sqlite3DBIModule: public Module
{
public:
   Sqlite3DBIModule();
   virtual ~Sqlite3DBIModule();

   virtual void onImportResolved( ImportDef* id, Symbol* sym, Item* value );
   virtual void onLinkComplete( VMContext* ctx );
   Class* classSql3liteDBIHandle() const { return m_classSql3liteDBIHandle; }

   virtual Service* createService( const String& name );
private:
   Class* m_dbiHandle;
   Class* m_classSql3liteDBIHandle;

   class Sqlite3Service: public DBIService {
   public:
      Sqlite3Service( Module* master );
      virtual ~Sqlite3Service();
      virtual DBIHandle *connect( const String &parameters );
   };
   Service* m_dbiService;
};

}

#endif

/* end of sqlite3_fm.h */
