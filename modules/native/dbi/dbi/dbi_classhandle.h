/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_classhandle.h
 *
 * DBI Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Tue, 21 Jan 2014 16:38:11 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>

#ifndef _FALCON_DBI_CLASSHANDLE_H_
#define _FALCON_DBI_CLASSHANDLE_H_

namespace Falcon {
namespace DBI {

class ClassHandle: public Class
{
public:
   ClassHandle();
   virtual ~ClassHandle();
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
};

}
}

#endif

/* end of dbi_classhandle.h */
