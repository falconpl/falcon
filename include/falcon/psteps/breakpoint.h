/*
   FALCON - The Falcon Programming Language.
   FILE: breakpoint.h

   Special statement -- breakpoint
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 22:20:48 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_BREAKPOINT_H
#define FALCON_BREAKPOINT_H

#include <falcon/statement.h>

namespace Falcon
{

/** Statement causing the VM to return.

 This is a debug feature that causes the VM to return from its main
 loop when it meets this statement.
 */
class FALCON_DYN_CLASS Breakpoint: public Statement
{
public:
   Breakpoint(int32 line=0, int32 chr = 0);
   virtual ~Breakpoint();

   void describeTo( String& tgt, int depth=0 ) const;

   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of breakpoint.h */
