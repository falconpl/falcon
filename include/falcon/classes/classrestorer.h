/*
   FALCON - The Falcon Programming Language.
   FILE: classrestorer.cpp

   Falcon engine -- class interfacing restorers in the VM.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 20:20:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CLASSRESTORER_H
#define FALCON_CLASSRESTORER_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/class.h>

#include <falcon/path.h>

namespace Falcon {

/**
 *    Class interfacing restorers in the VM.
 *
 */
class ClassRestorer: public Class
{
public:
   
   ClassRestorer();
   virtual ~ClassRestorer();
   
   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void* createInstance() const;
   
   //================================================
   // We're not using the usercarrier class
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   /*
    * TODO
   virtual void op_iter( VMContext* ctx, void* instance ) const;
   virtual void op_next( VMContext* ctx, void* instance ) const;
    */
};

}

#endif

/* end of classrestorer.h */
