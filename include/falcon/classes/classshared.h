/*
   FALCON - The Falcon Programming Language.
   FILE: classshared.h

   Interface for script to Shared variables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Nov 2012 12:52:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSSHARED_H_
#define _FALCON_CLASSSHARED_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{

class FALCON_DYN_CLASS ClassShared: public Class
{
public:
   ClassShared();
   virtual ~ClassShared();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;

   virtual void describe( void* instance, String& target, int, int ) const;
   virtual void gcMarkInstance( void* self, uint32 mark ) const;
   virtual bool gcCheckInstance( void* self, uint32 mark ) const;

   static void genericClassWait( const Class* childClass, VMContext* ctx, int32 pCount );
   static void genericClassTryWait( const Class* childClass, VMContext* ctx, int32 pCount );

protected:
   ClassShared( const String& name );
   ClassShared( const String& name, int64 type );
};

}

#endif

/* end of classshared.h */
