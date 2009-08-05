/*
   FALCON - The Falcon Programming Language.
   FILE: corerange.h

   Range object implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 07 Jan 2009 11:03:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_CORE_RANGE_H
#define FLC_CORE_RANGE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/garbageable.h>
#include <limits.h>

#ifndef LLONG_MIN
#define LLONG_MIN (-9223372036854775807LL-1)
#endif

namespace Falcon {

class FALCON_DYN_CLASS CoreRange: public Garbageable
{
   int64 m_start;
   int64 m_end;
   int64 m_step;

public:
   CoreRange():
      Garbageable(),
      m_start(0),
      m_end( 0 ),
      m_step( 0 )
   {}

   CoreRange( int64 start ):
      Garbageable(),
      m_start( start ),
      m_end( 0 ),
      m_step( LLONG_MIN )
   {}

   CoreRange( int64 start, int64 end ):
      Garbageable(),
      m_start( start ),
      m_end( end ),
      m_step( 0 )
   {}

   CoreRange( int64 start, int64 end, int64 step ):
      Garbageable(),
      m_start( start ),
      m_end( end ),
      m_step( step )
   {}

   CoreRange( const CoreRange &other ):
      Garbageable(),
      m_start( other.m_start ),
      m_end( other.m_end ),
      m_step( other.m_step )
   {}

   virtual ~CoreRange() {}

   bool isOpen() const { return m_step == (int64) LLONG_MIN; }
   int64 start() const { return m_start; }
   int64 end() const { return m_end; }
   int64 step() const { return m_step; }

   void setOpen() { m_step = LLONG_MIN; }
   void start( int64 s ) { m_start = s; }
   void end( int64 s ) { m_end = s; }
   void step( int64 s ) { m_step = s; }

   CoreRange* clone() const { return new CoreRange( *this ); }
};

}

#endif

/* end of corerange.h */

