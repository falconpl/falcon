/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_driverclass.h
 *
 * Base class for DBIHandle classes.
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Wed Jan 02 16:47:15 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/setup.h>
#include <falcon/types.h>

#ifndef FALCON_DBI_DRIVERCLASS_H
#define FALCON_DBI_DRIVERCLASS_H

#include <falcon/class.h>
namespace Falcon
{

/** Base class for DBIHandle classes.
 *
 * This groups common behavior of DBI database-specific modules.
 * Specific DBI database handlers have to directlu subclass this.
 *
 * Mainly, the only method that is to be implemented is the
 * createInstance() method, that should return a DBIHandle able to
 * deal with the target database system.
 */
class ClassDriverDBIHandle: public Class
{
public:
   ClassDriverDBIHandle( const String& name );
   virtual ~ClassDriverDBIHandle();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
};

}

#endif /* FALCON_DBI_DRIVERCLASS_H */

/* end of dbi_driverclass.h */

