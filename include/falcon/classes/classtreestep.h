/*
   FALCON - The Falcon Programming Language.
   FILE: classtreestep.h

   Class handling basic TreeStep common behavior.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 09:40:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSTREESTEP_H_
#define _FALCON_CLASSTREESTEP_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/class.h>
#include <falcon/function.h>

namespace Falcon {

class TreeStep;

/** Base handler class for TreeStep reflected at script level.
 
 */
class ClassTreeStep: public Class
{
public:
   ClassTreeStep();
   virtual ~ClassTreeStep();         
  

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* insatnce ) const;
   virtual void* createInstance() const;
   
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;   

   /** Standard behavior is that to let through without interfering.
    */
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;

   /** Standard behavior is that to let through without interfering.
   */
   virtual void restore( VMContext* ctx, DataReader* stream ) const;
   
   /** The normal behavior is that to flatten all via nth.
   */
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   
   /** The normal behavior is that to unflatten all via nth.
   */
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   
   virtual void enumerateProperties( void* instance, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* instance, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;
   
   virtual void op_call( VMContext* ctx, int32 paramCount, void* instance ) const;
   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* instance, const String& prop ) const;   
   virtual void op_getIndex(VMContext* vm, void* instance ) const;
   virtual void op_setIndex(VMContext* vm, void* instance ) const;

   virtual void op_iter( VMContext* ctx, void* instance ) const;
   virtual void op_next( VMContext* ctx, void* instance ) const;
   
   // This class is always alive as it resides in core/engine
   virtual bool gcCheckMyself( uint32 mark );

private:
   
   class InsertMethod: public Function {
   public:
      InsertMethod();
      virtual ~InsertMethod();
      void invoke( VMContext* ctx, int32 pCount = 0 );
   };

   class RemoveMethod: public Function {
   public:
      RemoveMethod();
      virtual ~RemoveMethod();
      void invoke( VMContext* ctx, int32 pCount = 0 );
   };

   class AppendMethod: public Function {
   public:
      AppendMethod();
      virtual ~AppendMethod();
      void invoke( VMContext* ctx, int32 pCount = 0 );
   };

   mutable InsertMethod m_insertMethod;
   mutable RemoveMethod m_removeMethod;
   mutable AppendMethod m_appendMethod;
};

}
#endif 

/* end of classtreestep.h */
