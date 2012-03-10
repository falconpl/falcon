/*
   FALCON - The Falcon Programming Language.
   FILE: bom.h

   Basic object model support.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 18:17:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_BOM_H
#define FALCON_BOM_H

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

class String;
class VMContext;
class Class;

/** Basic object model support.

 This class is used to store a dictionary of handlers for the basic object
 model methods.

 BOM methods are available to all the falcon items independently from their
 type. Although overridable by classes and pseudo-objects of various kinds
 (user-deined, blessed dictionaries and so on), they are used as fallback
 when a property is not found elsewhere.

 The BOM class is used to create a singleton in the engine, and is then used
 by the base Class::op_getProperty method.
 */
class BOM
{
public:
   BOM();
   virtual ~BOM();

   typedef void (*handler)(VMContext* ctx, const Class* cls, void* data);

   /** Gets a BOM handler.
    The handler takes care of getting the op_getProperty parameters and
    stroing the values.
    */
   handler get( const String& property ) const;

private:
   class Private;
   Private* _p;
};

/** Namespace holding the BOM property handlers.

 Names ending with _ are used as immediate properties (returning the required 
 value), while names without _ are used to return a method.
 */
namespace BOMH
{
void len(VMContext* ctx, const Class* cls, void* data);
void len_(VMContext* ctx, const Class* cls, void* data);

void baseClass(VMContext* ctx, const Class* cls, void* data);
void baseClass_(VMContext* ctx, const Class* cls, void* data);
void bound(VMContext* ctx, const Class* cls, void* data);
void bound_(VMContext* ctx, const Class* cls, void* data);
void className(VMContext* ctx, const Class* cls, void* data);
void className_(VMContext* ctx, const Class* cls, void* data);
void clone(VMContext* ctx, const Class* cls, void* data);
void clone_(VMContext* ctx, const Class* cls, void* data);
void describe(VMContext* ctx, const Class* cls, void* data);
void describe_(VMContext* ctx, const Class* cls, void* data);
void isCallable(VMContext* ctx, const Class* cls, void* data);
void isCallable_(VMContext* ctx, const Class* cls, void* data);
void ptr(VMContext* ctx, const Class* cls, void* data);
void ptr_(VMContext* ctx, const Class* cls, void* data);
void toString(VMContext* ctx, const Class* cls, void* data);
void toString_(VMContext* ctx, const Class* cls, void* data);
void typeId(VMContext* ctx, const Class* cls, void* data);
void typeId_(VMContext* ctx, const Class* cls, void* data);

// Theese are proper functions
void compare(VMContext* ctx, const Class* cls, void* data);
void derivedFrom(VMContext* ctx, const Class* cls, void* data);
}

}

#endif

/* end of bom.h */