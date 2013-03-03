/*
   FALCON - The Falcon Programming Language.
   FILE: classrequirement.h

   Requirement object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 22 Feb 2012 19:50:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSREQUIREMENT_H_
#define _FALCON_CLASSREQUIREMENT_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{

/** Purely abstract class used to store requirements in modules.
 
 This class is used by modules to use the flattening system on
 requirements.
 
 This is a pure virtual class; must be made concrete by adding a 
 "restore" that is able to create a real Requirement entity.
 */

class FALCON_DYN_CLASS ClassRequirement: public Class
{
public:

   ClassRequirement( const String& name );
   virtual ~ClassRequirement();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream, void*& empty ) const;
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;
};

}

#endif

/* end of classrequirement.h */
