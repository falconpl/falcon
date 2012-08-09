/*
   FALCON - The Falcon Programming Language.
   FILE: atomic_gcc.h

   Multithreaded extensions -- atomic operations for GCC compiler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 09 Aug 2012 00:01:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_ATOMIC_GCC_H
#define FALCON_ATOMIC_GCC_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <atomic>

namespace Falcon {
/**  An alias to the atomic integer type.
 */
typedef std::atomic_int atomic_int;

/** Performs an atomic thread safe increment. */
inline int32 atomicInc( atomic_int& atomic )
{
   return ++atomic;
}

/** Performs an atomic thread safe decrement. */
inline int32 atomicDec( atomic_int& atomic )
{
   return --atomic;
}

/** Performs an atomic thread safe addition. */
inline int32 atomicAdd( atomic_int& atomic, int32 value )
{
   return std::atomic_fetch_add(&atomic, value);
}

/** Perform a threadsafe fetch */
inline int32 atomicFetch( const atomic_int& atomic ) {
   return atomic.load(std::memory_order_acquire);
}

/** Perform a threadsafe fetch */
inline void atomicSet( atomic_int& atomic, int32 value ) {
   atomic = value;
}

inline int32 atomicExchange( atomic_int& atomic1, int32 value )
{
   return atomic1.exchange(value);
}

inline bool atomicCAS( atomic_int& target, int32 compareTo, int32 newVal )
{
   return target.compare_exchange_strong( compareTo, newVal );
}

inline bool atomicXor( atomic_int& target, int32 value )
{
   return target ^= value;
}

inline bool atomicAnd( atomic_int& target, int32 value )
{
   return target &= value;
}

inline bool atomicOr( atomic_int& target, int32 value )
{
   return target |= value;
}


}

#endif

/* end of atomic_gcc.h */
