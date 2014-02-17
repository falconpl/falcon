/*
   FALCON - The Falcon Programming Language.
   FILE: uri.h

   Falcon core module -- Interface to URI.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 20:20:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_URI_H
#define FALCON_CORE_URI_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/class.h>
#include <falcon/uri.h>
#include <falcon/path.h>

namespace Falcon {

class FALCON_DYN_CLASS ClassURI: public Class
{
public:
   
   ClassURI();
   virtual ~ClassURI();
    
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   virtual void store( VMContext*, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext*, DataReader* stream ) const;

   //=============================================================
   //
   virtual void* createInstance() const;   
   virtual bool op_init( VMContext* ctx, void* instance, int pcount ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;
   
};

}

#endif	/* FALCON_CORE_TOSTRING_H */

/* end of uri.h */
