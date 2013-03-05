/*
   FALCON - The Falcon Programming Language.
   FILE: classmessagequeue.h

   Message queue reflection in scripts
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 20 Feb 2013 14:15:46 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CLASSMESSAGEQUEUE_H
#define FALCON_CLASSMESSAGEQUEUE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/shared.h>
#include <falcon/classes/classshared.h>

namespace Falcon {

/*#
 @class MessageQueue
 @brief Multiple receivers message queue.

*/
class FALCON_DYN_CLASS ClassMessageQueue: public ClassShared
{
public:
   ClassMessageQueue();
   virtual ~ClassMessageQueue();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;
};

}

#endif	

/* end of classmessagequeue.h */
