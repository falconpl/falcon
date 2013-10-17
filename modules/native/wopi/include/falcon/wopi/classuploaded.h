/*
   FALCON - The Falcon Programming Language.
   FILE: classuploaded.h

   Web Oriented Programming Interface

   Interface to entities representing uploaded files.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Apr 2010 11:24:16 -0700

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_WOPI_CLASSUPLOADED_H_
#define _FALCON_WOPI_CLASSUPLOADED_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon {
namespace WOPI {

/** Interface to entities representing uploaded files.
 */
class ClassUploaded: public Class
{
public:
   ClassUploaded();
   virtual ~ClassUploaded();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
};

}
}

#endif

/* end of uploaded_ext.h */
