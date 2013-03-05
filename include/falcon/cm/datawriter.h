/*
   FALCON - The Falcon Programming Language.
   FILE: datawriter.h

   Falcon core module -- Wrapping for DataWriter
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 11 Dec 2011 17:32:58 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_CORE_DATAWRITER_H
#define FALCON_CORE_DATAWRITER_H

#include <falcon/types.h>
#include <falcon/class.h>

namespace Falcon {
namespace Ext {

class ClassDataWriter: public Class
{
public:
   ClassDataWriter( Class* clsStream );
   virtual ~ClassDataWriter();
   
   //=============================================================
   // Using a different carrier.
   
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* insatnce ) const;
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
   
   //=============================================================
   //

   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void* instance, int pcount ) const;
   
private:
   Class* m_clsStream;
};

}
}

#endif	/* FALCON_CORE_DATAWRITER_H */

/* end of datawriter.h */
