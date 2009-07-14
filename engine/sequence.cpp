/*
   FALCON - The Falcon Programming Language.
   FILE: sequence.cpp

   Definition of abstract sequence class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 13 Jul 2009 23:00:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/sequence.h>
#include <falcon/vm.h>
#include <falcon/error.h>

namespace Falcon {

inline void s_appendMe( VMachine *vm, Sequence* me, const Item &source, const Item &filter )
{
   if( filter.isNil() )
   {
      me->append( source );
   }
   else
   {
      vm->pushParameter( source );
      vm->callItemAtomic(filter,1);
      if ( ! vm->regA().isOob() )
         me->append( vm->regA() );
   }
}

void Sequence::comprehension( VMachine* vm, const Item& cmp, const Item& filter )
{
   if( cmp.isRange() )
   {
      {
         if ( cmp.asRangeIsOpen() )
         {
            throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .origin( e_orig_runtime )
               .extra( "open range" ) );
         }

         int64 start = cmp.asRangeStart();
         int64 end = cmp.asRangeEnd();
         int64 step = cmp.asRangeStep();
         if ( start == end ) {
            if ( step < 0 )
            {
               s_appendMe( vm, this, start, filter );
            }
            return;
         }

         if( start < end )
         {
            if ( step < 0 )
               return;
            if ( step == 0 )
               step = 1;

            while( start < end )
            {
               s_appendMe( vm, this, start, filter );
               start += step;
            }
         }
         else {
            if ( step > 0 )
               return;
            if ( step == 0 )
               step = -1;

            while( start >= end )
            {
               s_appendMe( vm, this, start, filter );
               start += step;
            }
         }
      }
   }
   else if ( cmp.isCallable() )
   {
      while( true )
      {
         vm->callItemAtomic( cmp, 0 );
         if( vm->regA().isOob() && vm->regA().isInteger() && vm->regA().asInteger() == 0 )
         {
            return;
         }

         Item temp = vm->regA();
         s_appendMe( vm, this, temp, filter );
      }
   }
   // todo --- remove this as soon as we have iterators on ItemArrays
   else if ( cmp.isArray() )
   {
      const CoreArray& arr = *cmp.asArray();

      for( uint32 i = 0; i < arr.length(); i ++ )
      {
         s_appendMe( vm, this, arr[i], filter );
      }
   }
   else if ( (cmp.isObject() && cmp.asObjectSafe()->getFalconData()->isSequence()) )
   {
      //Sequence* seq = cmp.isArray() ? &cmp.asArray()->items() : cmp.asObjectSafe()->getSequence();

      Sequence* seq = cmp.asObjectSafe()->getSequence();
      CoreIterator *iter = seq->getIterator();
      try {
         while( iter->isValid() )
         {
            s_appendMe( vm, this, iter->getCurrent(), filter );
            iter->next();
         }
      }
      catch(...)
      {
         delete iter;
         throw;
      }

      delete iter;
   }
   else {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin( e_orig_runtime )
               .extra( "A|C|R|Sequence, [C]" ) );
   }
}

}

/* end of sequence.cpp */
