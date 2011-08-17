/*
   FALCON - The Falcon Programming Language.
   FILE: corefunc.h

   Abstract class for immediately callable items at language levels.
   They are functions and arrays. Classes, methods and class methods
   are secondary callable items, which relay on this primary callable
   items (arrays and functions).

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 28 Jul 2009 00:32:58 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CALLPOINT_H_
#define FALCON_CALLPOINT_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/garbageable.h>

namespace Falcon
{

class FALCON_DYN_CLASS CallPoint:  public Garbageable
{
public:
   virtual ~CallPoint() {}

   virtual const String& name() const = 0;
   virtual void readyFrame( VMachine* vm, uint32 paramCount ) = 0;
   virtual bool isFunc() const = 0;

};

}

#endif /* CALLPOINT_H_ */

/* end of callpoint.h */
