/*
   FALCON - The Falcon Programming Language.
   FILE: variable.cpp

   Pair of item value and location where the item is accounted in memory.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Jun 2012 15:33:16 +0200

   -------------------------------------------------------------------
(C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/variable.h>
#include <falcon/engine.h>
#include <falcon/collector.h>
#include <falcon/classes/classrawmem.h>

namespace Falcon {

void Variable::makeReference( Variable* original, Variable* copy )
{
   static Collector* coll = Engine::instance()->collector();
   static ClassRawMem* rawMem = Engine::instance()->rawMemClass();
   
   if( original->m_base != 0 ) {
      copy->m_base = original->m_base;
      copy->m_value = original->m_value;
   }
   else {
      // craete  a place in memory that is sure to be properly aligned
      Variable::t_aligned_variable *aligned_variable;      
      aligned_variable = (t_aligned_variable*) rawMem->allocate( sizeof(t_aligned_variable) );
      
      // get the pointers to this aligned memory
      copy->m_base = &aligned_variable->count;
      copy->m_value = &aligned_variable->value;
      // copy the original item.
      *copy->m_value = *original->m_value;
      
      // fix the original item to point to the new location.
      original->m_base = copy->m_base;
      original->m_value = copy->m_value;
      
      // assign to the collector.
      FALCON_GC_STORE(coll, rawMem, aligned_variable);
   }
}

}

/* end of variable.cpp */
