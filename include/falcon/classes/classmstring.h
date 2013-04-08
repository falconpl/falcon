/*
   FALCON - The Falcon Programming Language.
   FILE: classmstring.h

   Mutable String object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 08 Apr 2013 15:01:23 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSMSTRING_H_
#define _FALCON_CLASSMSTRING_H_

#include <falcon/setup.h>
#include <falcon/classes/classstring.h>

namespace Falcon
{
class PStep;

/**
 Class handling a string as an item in a falcon script.
 */

class FALCON_DYN_CLASS ClassMString: public ClassString
{
public:
   ClassMString();
   virtual ~ClassMString();
   
   virtual void describe( void* instance, String& target, int, int ) const;

   virtual void op_aadd( VMContext* ctx, void* self ) const;
   virtual void op_amul( VMContext* ctx, void* self ) const;
   virtual void op_adiv( VMContext* ctx, void* self ) const;

   virtual void op_setIndex( VMContext* ctx, void* self ) const;

};

}

#endif /* _FALCON_CLASSMSTRING_H_ */

/* end of classmstring.h */
