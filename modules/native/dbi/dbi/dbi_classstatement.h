/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_classstatement.h
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

#ifndef _FALCON_DBI_CLASSSTATEMENT_H_
#define _FALCON_DBI_CLASSSTATEMENT_H_

namespace Falcon {

class ClassStatement: public Class
{
public:
   ClassStatement();
   virtual ~ClassStatement();
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
};

}

#endif

/* dbi_classstatement.h */