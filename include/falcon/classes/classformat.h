/*
   FALCON - The Falcon Programming Language.
   FILE: classformat.h

   Format type handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 01 Feb 2013 23:31:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSFORMAT_H_
#define _FALCON_CLASSFORMAT_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/class.h>

namespace Falcon
{

/**
 Class explicitly holding a format.
 */

class FALCON_DYN_CLASS ClassFormat: public Class
{
public:

   ClassFormat();
   virtual ~ClassFormat();

   virtual int64 occupiedMemory( void* instance ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext*, DataWriter* dw, void* data ) const;
   virtual void restore( VMContext* , DataReader* dr ) const;

   virtual void describe( void* instance, String& target, int, int ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   //=============================================================
   virtual bool op_init( VMContext* ctx, void*, int32 pcount ) const;
   virtual void op_compare( VMContext* ctx, void* self ) const;
};

}

#endif /* _FALCON_CLASSFORMAT_H_ */

/* end of classformat.h */
