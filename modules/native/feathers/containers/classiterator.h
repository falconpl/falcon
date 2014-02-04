/* FALCON - The Falcon Programming Language.
 * FILE: classiterator.h
 *
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sat, 01 Feb 2014 12:56:12 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: The above AUTHOR
 *
 * See LICENSE file for licensing details.
 */

#ifndef _FALCON_FEATHERS_CONTAINERS_CLASSITERATOR_H_
#define _FALCON_FEATHERS_CONTAINERS_CLASSITERATOR_H_

#include <falcon/class.h>

namespace Falcon {

class ClassIterator: public Class
{
public:
   ClassIterator();
   virtual ~ClassIterator();

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
};

}

#endif

/* end of classiterator.h */
