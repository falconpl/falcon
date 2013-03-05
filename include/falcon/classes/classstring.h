/*
   FALCON - The Falcon Programming Language.
   FILE: classstring.h

   String object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 15:11:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSSTRING_H_
#define _FALCON_CLASSSTRING_H_

#include <falcon/setup.h>
#include <falcon/class.h>
#include <falcon/string.h>
#include <falcon/instancelock.h>

#include <falcon/pstep.h>
namespace Falcon
{

/**
 Class handling a string as an item in a falcon script.
 */

class FALCON_DYN_CLASS ClassString: public Class
{
public:

   ClassString();
   virtual ~ClassString();

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

   virtual void op_add( VMContext* ctx, void* self ) const;
   virtual void op_aadd( VMContext* ctx, void* self ) const;

   virtual void op_getIndex( VMContext* ctx, void* self ) const;
   virtual void op_setIndex( VMContext* ctx, void* self ) const;

   virtual void op_compare( VMContext* ctx, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;
   virtual void op_isTrue( VMContext* ctx, void* self ) const;

private:
   InstanceLock m_lock;

   class FALCON_DYN_CLASS NextOp: public PStep
   {
   public:
      NextOp( ClassString* owner );
      static void apply_( const PStep*, VMContext* vm );
   private:
      ClassString* m_owner;
   } m_nextOp;
   
   class FALCON_DYN_CLASS InitNext: public PStep
   {
   public:
      InitNext();
      static void apply_( const PStep*, VMContext* vm );
   } m_initNext;
};

}

#endif /* _FALCON_CLASSSTRING_H_ */

/* end of classstring.h */
