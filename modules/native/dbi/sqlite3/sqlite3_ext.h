/*
 * FALCON - The Falcon Programming Language.
 * FILE: sqlite3_ext.h
 *
 * SQLite3 Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 16:47:15 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/setup.h>
#include <falcon/types.h>

#ifndef FALCON_DBI_SQLITE3_EXT_H
#define FALCON_DBI_SQLITE3_EXT_H

#include <falcon/class.h>
namespace Falcon
{
namespace Ext
{

class ClassSqlite3DBIHandle: public Class
{
public:
   ClassSqlite3DBIHandle();
   virtual ~ClassSqlite3DBIHandle();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
};

}
}

#endif /* SQLITE3_EXT_H */

/* end of sqlite3_ext.h */

