/*
   FALCON - The Falcon Programming Language.
   FILE: classmantra.h

   Handler for generic mantra entities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 26 Feb 2012 00:30:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSMANTRA_H_
#define _FALCON_CLASSMANTRA_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{

class FALCON_DYN_CLASS ClassMantra: public Class
{
public:

   ClassMantra();
   virtual ~ClassMantra();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void describe( void* instance, String& target, int, int ) const;
   virtual void gcMarkInstance( void* self, uint32 mark ) const;
   virtual bool gcCheckInstance( void* self, uint32 mark ) const;
   
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream) const;
   // mantras have no flattening.
   
   //=============================================================

   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;

protected:
   ClassMantra( const String& name, int64 type );
   
};

}

#endif

/* end of classmantra.h */
