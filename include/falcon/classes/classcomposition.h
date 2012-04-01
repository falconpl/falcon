/*
   FALCON - The Falcon Programming Language.
   FILE: classcomposition.h

   A Functional composition. 
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Apr 2012 16:36:55 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_COMPOSITION_H
#define FALCON_COMPOSITION_H

#include <falcon/setup.h>
#include <falcon/pstep.h>
#include <falcon/class.h>

namespace Falcon {

/** A functional composition.
    Calling an instance of this class is equivalent to call the second
    operand and then the first with the result of the second.
 */
class ClassComposition: public Class
{
public:
   ClassComposition();   
   ~ClassComposition() {}
   
   virtual void* createInstance() const; 
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream, void*& empty ) const;
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
   virtual void enumerateProperties( void* instance, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* instance, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void op_call( VMContext* ctx, int32 paramCount, void* instance ) const;
   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* instance, const String& prop ) const;

   void* createComposition( const Item& first, const Item& second ) const;
   

private:
   
   class FALCON_DYN_CLASS ApplyFirst: public PStep {
   public:
      ApplyFirst();
      virtual ~ApplyFirst() {}
      static void apply_( const PStep*, VMContext* vm );
   } m_applyFirst;
};

}

#endif

/* end of classcomposition.h */
