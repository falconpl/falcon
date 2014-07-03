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
#include <falcon/class.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {
class Function_readItem;

class ClassDataReader: public Class
{
public:
   ClassDataReader( Class* clsStream );
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

   Class* m_clsStream;


   class FALCON_DYN_CLASS ReadItemNext: public PStep
   {
   public:
      ReadItemNext(ClassDataReader* owner): m_owner(owner) { apply = apply_; }
      virtual ~ReadItemNext() {}
      static void apply_( const PStep* ps, VMContext* ctx );

      //Need to do something about this
      ClassDataReader* m_owner;
   };

   friend class ReadItemNext;
   friend class Function_readItem;
   ReadItemNext m_readItemNext;
};

}
}

#endif	/* FALCON_CORE_DATAREADER_H */

/* end of datareader.h */
