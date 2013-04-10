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

   // exposing the instance lock
   InstanceLock::Token* lockInstance( void* instance ) const { return m_lock.lock(instance); }
   void unlockInstance( InstanceLock::Token* tk ) const { m_lock.unlock(tk); }

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
   // Immutable operators
   //
   virtual bool op_init( VMContext* ctx, void*, int32 pcount ) const;

   virtual void op_add( VMContext* ctx, void* self ) const;

   virtual void op_mul( VMContext* ctx, void* self ) const;
   virtual void op_div( VMContext* ctx, void* self ) const;
   virtual void op_mod( VMContext* ctx, void* self ) const;

   virtual void op_getIndex( VMContext* ctx, void* self ) const;

   virtual void op_compare( VMContext* ctx, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;
   virtual void op_isTrue( VMContext* ctx, void* self ) const;

   virtual void op_in( VMContext* ctx, void* self ) const;

   virtual void op_iter( VMContext* ctx, void* self ) const;
   virtual void op_next( VMContext* ctx, void* self ) const;

   //=============================================================
   // Mutable operators
   //
   virtual void op_aadd( VMContext* ctx, void* self ) const;
   virtual void op_amul( VMContext* ctx, void* self ) const;
   virtual void op_amod( VMContext* ctx, void* self ) const;

   virtual void op_setIndex( VMContext* ctx, void* self ) const;

   //=============================================================
   // Model strings
   //
   String* m_modelAllAlpha;
   String* m_modelAllAlphaNum;
   String* m_modelAllDigit;
   String* m_modelAllUpper;
   String* m_modelAllLower;
   String* m_modelAllPunct;

protected:
   ClassString( const String& subclassName );

   class PStepInitNext;
   PStep* m_initNext;

   class PStepNextOp;
   class PStep* m_nextOp;

   InstanceLock m_lock;

private:

   void init();
};

}

#endif /* _FALCON_CLASSSTRING_H_ */

/* end of classstring.h */
