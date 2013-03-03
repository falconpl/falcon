/*
   FALCON - The Falcon Programming Language.
   FILE: classclosedddata.h

   Handler for data in closure entities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 10 Jan 2013 15:02:40 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSCLOSEDDATA_H_
#define _FALCON_CLASSCLOSEDDATA_H_

#include <falcon/setup.h>
#include <falcon/class.h>

#include <falcon/pstep.h>

namespace Falcon
{

/** Handler for data in closure entities.
 *
 *  Not visible at script level.
 */
class FALCON_DYN_CLASS ClassClosedData: public Class
{
public:

   ClassClosedData();
   virtual ~ClassClosedData();
   
   //=============================================================

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   
   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   virtual void gcMarkInstance( void* self, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
};

}

#endif /* _FALCON_CLASSARRAY_H_ */

/* end of classclosedddata.h */
