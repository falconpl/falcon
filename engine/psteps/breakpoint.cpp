/*
   FALCON - The Falcon Programming Language.
   FILE: breakpoint.cpp

   Special statement -- breakpoint
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:51:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/ps/breakpoint.cpp"

#include <falcon/trace.h>
#include <falcon/psteps/breakpoint.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include <falcon/textwriter.h>

namespace Falcon
{

Breakpoint::Breakpoint( int32 line, int32 chr ):
   Statement( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_breakpoint )
   apply = apply_;
}

Breakpoint::Breakpoint( const Breakpoint& other ):
   Statement(other)
{
}

Breakpoint::~Breakpoint()
{
}
   
Breakpoint* Breakpoint::clone() const
{
   return new Breakpoint(*this);
}


void Breakpoint::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );
   tw->write( "/* breakpoint */\n" );
}


void Breakpoint::apply_( const PStep*, VMContext* ctx )
{
   ctx->setBreakpointEvent();
   ctx->popCode();
}

}

/* end of breakpoint.cpp */
