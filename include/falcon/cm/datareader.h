/*
   FALCON - The Falcon Programming Language.
   FILE: datareader.h

   Falcon core module -- Wrapping for DataReader
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 11 Dec 2011 17:32:58 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_DATAREADER_H
#define FALCON_CORE_DATAREADER_H

#include <falcon/types.h>
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/classes/classuser.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

class ClassDataReader: public ClassUser
{
public:
   ClassDataReader();
   virtual ~ClassDataReader();
   
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
   
   //====================================================
   // Properties.
   //
   
   FALCON_DECLARE_PROPERTY( endianity );
   FALCON_DECLARE_PROPERTY( sysEndianity );
   
   FALCON_DECLARE_METHOD( read, "data:S|M, count:[N]" );
   FALCON_DECLARE_METHOD( readBool, "" );
   FALCON_DECLARE_METHOD( readChar, "" );
   FALCON_DECLARE_METHOD( readByte, "" );
   FALCON_DECLARE_METHOD( readI16, "" );
   FALCON_DECLARE_METHOD( readU16, "" );
   FALCON_DECLARE_METHOD( readI32, "" );
   FALCON_DECLARE_METHOD( readU32, "" );
   FALCON_DECLARE_METHOD( readI64, "" );
   FALCON_DECLARE_METHOD( readU64, "" );
   FALCON_DECLARE_METHOD( readF32, "" );
   FALCON_DECLARE_METHOD( readF64, "" );
   FALCON_DECLARE_METHOD( readString, "" );
   FALCON_DECLARE_METHOD( readItem, "model:Class" );
   
   FALCON_DECLARE_METHOD( sync, "" );
   FALCON_DECLARE_METHOD( eof, "" );

   
   class FALCON_DYN_CLASS ReadItemNext: public PStep
   {
   public:
      ReadItemNext(ClassDataReader* owner): m_owner(owner) { apply = apply_; }
      virtual ~ReadItemNext() {}
      static void apply_( const PStep* ps, VMContext* ctx );
   private:
      ClassDataReader* m_owner; 
   };

   friend class ReadItemNext;
   ReadItemNext m_readItemNext;
};

}
}

#endif	/* FALCON_CORE_DATAREADER_H */

/* end of datareader.h */
