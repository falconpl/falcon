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
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/classes/classuser.h>

namespace Falcon {
namespace Ext {

class ClassDataWriter: public ClassUser
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
   
   //====================================================
   // Properties.
   //
   
   FALCON_DECLARE_PROPERTY( endianity );
   FALCON_DECLARE_PROPERTY( sysEndianity );
   
   FALCON_DECLARE_METHOD( write, "data:S|M, count:[N], start:[N]" );
   FALCON_DECLARE_METHOD( writeBool, "data:B" );
   FALCON_DECLARE_METHOD( writeChar, "data:S" );
   FALCON_DECLARE_METHOD( writeByte, "data:N" );
   FALCON_DECLARE_METHOD( writeI16, "data:N" );
   FALCON_DECLARE_METHOD( writeU16, "data:N" );
   FALCON_DECLARE_METHOD( writeI32, "data:N" );
   FALCON_DECLARE_METHOD( writeU32, "data:N" );
   FALCON_DECLARE_METHOD( writeI64, "data:N" );
   FALCON_DECLARE_METHOD( writeU64, "data:N" );
   FALCON_DECLARE_METHOD( writeF32, "data:N" );
   FALCON_DECLARE_METHOD( writeF64, "data:N" );
   FALCON_DECLARE_METHOD( writeString, "data:S" );
   FALCON_DECLARE_METHOD( writeItem, "data:X" );
   
   FALCON_DECLARE_METHOD( flush, "" );
};

}
}

#endif	/* FALCON_CORE_DATAWRITER_H */

/* end of datawriter.h */
