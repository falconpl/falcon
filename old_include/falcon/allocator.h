/*
   FALCON - The Falcon Programming Language.
   FILE: flc_allocator.h

   Standard falcon allocator.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef flc_ALLOCATOR_H
#define flc_ALLOCATOR_H

#include <falcon/memory.h>

namespace Falcon {

/** Standard falcon allocator.
   Bridges STL with falcon memory allocation.
*/

template<class _Tp> 
class Allocator
{
public:
      typedef size_t     size_type;
      typedef ptrdiff_t  difference_type;
      typedef _Tp*       pointer;
      typedef const _Tp* const_pointer;
      typedef _Tp&       reference;
      typedef const _Tp& const_reference;
      typedef _Tp        value_type;

      template<typename _Tp1>
        struct rebind
        { typedef Allocator<_Tp1> other; };

      Allocator() throw() {}
      //Allocator(const Allocator&) throw() {}
      template<typename _Tp1>
        Allocator(const Allocator<_Tp1>&) throw() {}
      ~Allocator() throw() {}

      pointer
      address(reference __x) const { return &__x; }

      const_pointer
      address(const_reference __x) const { return &__x; }

      size_type max_size() const throw() { return size_t(-1) / sizeof(_Tp); }

      void construct(pointer __p, const _Tp& __val) { new(__p) _Tp(__val); }
      void destroy(pointer __p) { __p->~_Tp(); }


   _Tp* allocate( size_type n, const void* = 0 )
   {
      return reinterpret_cast<_Tp*>( memAlloc( n * sizeof( _Tp ) ) );
   }

   void deallocate( _Tp* p, size_type n )
   {
      memFree( p );
   }

#ifdef _MSC_VER
   void deallocate( void* p, size_type n )
   {
      memFree( p );
   }
   _Tp* _Charalloc( size_type n )
   {
      return allocate( n );
   }
#endif
};

template <class T1, class T2>
bool operator== (const Allocator<T1>& one,
               const Allocator<T2>& two) throw() {
   if ( &one == &two ) return true;
   return false;
}

template <class T1, class T2>
bool operator!= (const Allocator<T1>& one,
               const Allocator<T2>& two) throw() {
   if ( &one == &two ) return false;
}

}

#endif
/* end of flc_allocator.h */
