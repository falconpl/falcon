/*
   FALCON - The Falcon Programming Language.
   FILE: classmodule.h

   Module object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 22 Feb 2012 19:50:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSMODULE_H_
#define _FALCON_CLASSMODULE_H_

#include <falcon/setup.h>
#include <falcon/class.h>
#include <falcon/function.h>

namespace Falcon
{

/*#
 * @class Module
 * @brief Reflection of Falcon module.
 *
 * @prop name The logical name of the module.
 * @prop uri A string containing the full URI where this module
 *       is located. Can be empty if the module wasn't created
 *       from a serialized resource.
 */

class FALCON_DYN_CLASS ClassModule: public Class
{
public:

   ClassModule();
   virtual ~ClassModule();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;

   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream ) const;
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

private:
   void restoreModule( Module* mod, DataReader* stream ) const;
};

}

#endif

/* end of classmodule.h */
