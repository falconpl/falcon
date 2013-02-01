/*
   FALCON - The Falcon Programming Language.
   FILE: atomic_win.h

   Multithreaded extensions -- atomic operations for Windows.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai, Paul Davey
   Begin: Thu, 09 Aug 2012 00:01:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_ATOMIC_WIN_H
#define FALCON_ATOMIC_WIN_H

#include <falcon/setup.h>
#include <falcon/types.h>

#include <windows.h>

namespace Falcon {
/**  An alias to the atomic integer type.
 */
typedef LONG atomic_int;

/** Performs an atomic thread safe increment. */
inline int32 atomicInc( atomic_int& atomic )
{
   return InterlockedIncrement( &atomic );
}

/** Performs an atomic thread safe decrement. */
inline int32 atomicDec( atomic_int& atomic )
{
   return InterlockedDecrement( &atomic );
}

/** Performs an atomic thread safe addition. */
inline int32 atomicAdd( atomic_int& atomic, atomic_int value )
{
   return InterlockedAdd( &atomic, value );
}

/** Perform a threadsafe fetch */
inline int32 atomicFetch( const atomic_int& atomic ) {
   return InterlockedCompareExchange( (volatile LONG*)&atomic, 0, 0 );
}

/** Perform a threadsafe set.*/
inline void atomicSet( atomic_int& atomic, atomic_int value ) {
   InterlockedExchange( &atomic, value );
}

/** Sets the given value in atomic, and returns the previous value. */
inline atomic_int atomicExchange( atomic_int& atomic, atomic_int value )
{
   return InterlockedExchange( &atomic, value );
}

inline bool atomicCAS( atomic_int& target, atomic_int compareTo, atomic_int newVal )
{
   return InterlockedCompareExchange( &target, newVal, compareTo );
}

inline atomic_int atomicXor( atomic_int& target, atomic_int value )
{
   return InterlockedXor(&target, value);
}

inline atomic_int atomicAnd( atomic_int& target, atomic_int value )
{
   return InterlockedAnd(&target, value);
}

inline atomic_int atomicOr( atomic_int& target, atomic_int value )
{
   return InterlockedOr(&target, value);
}


}

#endif

/* end of atomic_win.h */
