/*
   FALCON - The Falcon Programming Language.
   FILE: coreclass.h

   Class defined by a Falcon script.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 16:04:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CORECLASS_H_
#define _FALCON_CORECLASS_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{

/** Class defined by a Falcon script.

 This class implements a class as seen by a Falcon script. It stores the
 properties and the methods as declared by the script and has support to allow
 the script to re-define autonomously the operands by declaring special methods
 at script level.

 The CoreClass Class has meta-information about the objects it can create
 and about the basic, default properties it provides.

 Notice that it has also implicit properties as "name",
 */
class FALCON_DYN_CLASS CoreClass: public Class
{
public:

   CoreClass();
   virtual ~CoreClass();

   bool isCoreClass();

   virtual void* create( void* creationParams ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   virtual void describe( void* instance, String& target ) const;

   //=============================================================

   virtual void op_isTrue( VMachine *vm, void* self ) const;
};

}

#endif /* _FALCON_CORECLASS_H_ */

/* end of coreclass.h */
