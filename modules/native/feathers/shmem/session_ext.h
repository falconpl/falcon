/*
   FALCON - The Falcon Programming Language.
   FILE: session_ext.h

   Falcon script interface for Inter-process persistent data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 15 Nov 2013 15:31:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_FEATHERS_SESSION_EXT_H
#define FALCON_FEATHERS_SESSION_EXT_H

#include <falcon/class.h>

namespace Falcon {

class ClassSession: public Class
{
public:
   ClassSession();
   virtual ~ClassSession();

   virtual int64 occupiedMemory( void* instance ) const;
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
};

}

#endif

/* end of session_ext.h */
