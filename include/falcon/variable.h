/*
   FALCON - The Falcon Programming Language.
   FILE: variable.h

   Pair of item value and location where the item is accounted in memory.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Jun 2012 15:33:16 +0200

   -------------------------------------------------------------------
(C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_VARIABLE_H
#define FALCON_VARIABLE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/item.h>

namespace Falcon {


/**
Pair of item value and location where the item is accounted in memory.
*/
class FALCON_DYN_CLASS Variable {

public:
   /** Memory marker for aligned variables.
    
    This structure just represents a pair of GC data and item data
    that can be placed anywhere in memory.
    
    By using this definition to create and access a dynamic memory area,
    the final code is ensured to be properly aligned on any architecture,
    and yet starting with an uint32 which is the data used by ClassRawMem
    to account for unknown foreign memory.
    */ 
   typedef struct {
         uint32 count;
         Item value;
   }  t_aligned_variable;

   inline Variable():
      m_value( 0 ),
      m_base( 0 )
   {}

   inline Variable( uint32* base, Item* value ):
      m_value( value ),
      m_base( base )
   {}
   
   inline Item* value() const { return m_value; }
   inline void value( Item* v ) { m_value = v; }
   
   inline uint32* base() const { return m_base; }
   inline void base( uint32* v ) { m_base = v; }
   
   inline void gcMark( uint32 mark ) {
      if( m_base != 0 ) *m_base = mark;
   }
   
   inline uint32 gcMark() const { return *m_base; }

   inline void set( uint32* base, Item* value ) {
      m_base = base;
      m_value = value;
   }
   
   bool isReference() {
      return m_base != 0;
   }
   
   /** Creates a reference to the target variable.
    \param original The variable to be referenced.
    
    If the variable is already a reference, it is just copied,
    otherwise a new memory area is created and assigned to the GC.
    
    When complete, both the original and the copy are pointing to the same
    objects.
    */
   inline void makeReference( Variable* original ) { makeReference( original, this ); }
   
   
   /** Creates a reference to the target variable.
    \param original The variable to be referenced.
    \param copy The variable where the reference is stored.
    
    If the variable is already a reference, it is just copied,
    otherwise a new memory area is created and assigned to the GC.
    
    When complete, both the original and the copy are pointing to the same
    objects.
    */
   static void makeReference( Variable* original, Variable* copy );
      
   /** Creates a garbage collected variable that is free in memory.
    \param var a place where A GC allocated and aligned variable ready for use is stored.
    */
   static void makeFreeVariable( Variable& var );

private:
   Item* m_value;
   uint32* m_base;
};
}

#endif	/* FALCON_VARIABLE_H */

/* end of variable.h */
