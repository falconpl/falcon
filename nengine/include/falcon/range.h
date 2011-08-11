/*
   FALCON - The Falcon Programming Language.
   FILE: range.h

   Range object implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai & Paul Davey
   Begin: Wed, 27 Jul 2011 11:26:00 +1200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_RANGE_H
#define FLC_RANGE_H

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

class FALCON_DYN_CLASS Range
{
   int64 m_start;
   int64 m_end;
   int64 m_step;
   int32 m_gcMark;
   bool m_open;
   

public:
   Range():
      m_start( 0 ),
      m_end( 0 ),
      m_step( 0 ),
      m_open( false ),
      m_gcMark( 0 )
   {}

   Range( int64 start ):
      m_start( start ),
      m_end( 0 ),
      m_step( 0 ),
      m_open( true ),
      m_gcMark( 0 )
   {}

   Range( int64 start, int64 end ):
      m_start( start ),
      m_end( end ),
      m_step( 0 ),
      m_open( false ),
      m_gcMark( 0 )
   {}

   Range( int64 start, int64 end, int64 step ):
      m_start( start ),
      m_end( end ),
      m_step( step ),
      m_open( false ),
      m_gcMark( 0 )
   {}
   
   Range( int64 start, int64 end, int64 step, bool open ):
      m_start( start ),
      m_end( end ),
      m_step( step ),
      m_open( open ),
      m_gcMark( 0 )
   {}

   Range( const Range &other ):
      m_start( other.m_start ),
      m_end( other.m_end ),
      m_step( other.m_step ),
      m_open( other.m_open ),
      m_gcMark( 0 )
   {}

   virtual ~Range() {}

   inline bool isOpen() const { return m_open; }
   inline int64 start() const { return m_start; }
   inline int64 end() const { return m_end; }
   inline int64 step() const { return m_step; }
   inline int32 gcMark() const { return m_gcMark; }

   inline void setOpen(bool open) { m_open = open; }
   inline void start( int64 s ) { m_start = s; }
   inline void end( int64 s ) { m_end = s; }
   inline void step( int64 s ) { m_step = s; }
   inline void gcMark( int32 mark ) { m_gcMark = mark; }

};

}

#endif

/* end of range.h */

