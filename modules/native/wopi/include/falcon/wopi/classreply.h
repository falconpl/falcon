/*
   FALCON - The Falcon Programming Language.
   FILE: classreply.h

   Falcon Web Oriented Programming Interface.

   Interface to Reply object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 16 Oct 2013 12:38:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_WOPI_CLASSREPLY_H
#define FALCON_WOPI_CLASSREPLY_H

#include <falcon/module.h>
#include <falcon/class.h>

namespace Falcon{
namespace WOPI {

//#define FALCON_WOPI_REQUEST_GETS_PROP           "gets"


class ClassReply: public Class
{
public:
   ClassReply();
   virtual ~ClassReply();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
};

}
}

#endif

/* end of classreply.h */
