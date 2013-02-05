/*
   FALCON - The Falcon Programming Language.
   FILE: classmodspace.h

   Handler for dynamically created module spaces.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 05 Feb 2013 18:07:35 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CLASSMODSPACE_H
#define FALCON_CLASSMODSPACE_H

#include <falcon/setup.h>
#include <falcon/pstep.h>
#include <falcon/class.h>

namespace Falcon {

/** Handler for dynamically created module spaces.
 */
class ClassModSpace: public Class
{
public:
   ClassModSpace();
   ~ClassModSpace();
   
   virtual void* createInstance() const; 
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   
   // module spaces are not serializable; the module they hold
   // have the ability to re-create them on need.

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   /*
   virtual void enumerateProperties( void* instance, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* instance, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* instance, const String& prop ) const;
    */
private:

};

}

#endif

/* end of classmodspace.h */
