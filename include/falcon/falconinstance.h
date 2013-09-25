/*
   FALCON - The Falcon Programming Language.
   FILE: falconinstance.h

   Instance of classes declared in falcon scripts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jun 2011 14:35:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_INSTANCE_H_
#define _FALCON_INSTANCE_H_

#include <falcon/setup.h>
#include <falcon/itemarray.h>
#include <falcon/delegatemap.h>

namespace Falcon
{

class FalconClass;
class CoreInstance;

/** Instance of classes declared in falcon scripts.

 Instances of FalconClass entities (or, basic Falcon objects) are
 just a set of values stored in an array plus a reference to
 the FalconClass that describes the instance structure.

 Access is normally meant to be performed directly by the virtual machine,
 and so, errors in access cause directly an AccessError raising.

 \note This class is fully non-virtual.
 */
class FALCON_DYN_CLASS FalconInstance
{
public:
   /** Creates the instance.
    \param origin The class from which this object originates.
    */
   FalconInstance( const FalconClass* origin );
   FalconInstance( const FalconInstance& other );
   ~FalconInstance();

   /** The originator class of this instance. */
   const FalconClass* origin() const;

   /** Gets the value of a member of this class.
    
    \param name The name of the property to be queried.
    \param target The item where to store the value of the property
    \thorw AccessError if the property is not found.

    This method fills the target item with a property value or with a method
    taken from the origin class.

    As this is an advanced method, usually called directly by the VM or by
    code serving as VM operation proxy, if the property is not found an
    error is immediately raised. To determine what properties are available,
    access directly the origin class (and consider using the data() array).

    \note Target is not locked while copy is performed. In case the target
    is to be sent to an unknown item storage, use a local storage and then perform
    a correctly interlocked copy across the local temporary storage and the
    remote target (or, if it's viable, lock the target prior invockin this method).

    \see FalconClass

    */
   bool getProperty( const String& name, Item& target ) const;

   /** Sets the value of a property.
    \param name The name of the property to be queried.
    \param target The item where to store the value of the property
    \thorw AccessError if the property is not found.
    \thorw AccessTypeError if the property is found, but is read-only.

    This method changes the value of a property in the class. If the
    property name represents a base class, a method or a state, a read-only
    error is thrown.

    As this is an advanced method, usually called directly by the VM or by
    code serving as VM operation proxy, if the property is not found an
    error is immediately raised. To determine what properties are available,
    access directly the origin class (and consider using the data() array).

    \note Target is not locked while copy is performed. In case the target
    is to be sent to an unknown item storage, use a local storage and then perform
    a correctly interlocked copy across the local temporary storage and the
    remote target (or, if it's viable, lock the target prior invockin this method).

    \see FalconClass

    */
   bool setProperty( const String& name, const Item& value );


   /** Clone this instance.
    \return A flat copy of this instance.    
    \note all the items in the data array are marked as copied.
    */
   FalconInstance* clone() const { return new FalconInstance(*this); }

   /** Marks this instance. 
    \param mark The GC Mark ID.
    */
   void gcMark( uint32 mark );
   
   uint32 currentMark() const;

private:
   FalconInstance();

   Item* getProperty_internal( const String* name ) const;
   void makeStorageData( ItemArray& array ) const;
   void restoreFromStorageData( ItemArray& array );

   // IMPORTANT
   // Use single pointer implementation semantic; never add a visible property!
   class Private;
   Private* _p;
   
   friend class FalconClass;
};

}

#endif /* _FALCON_INSTANCE_H_ */

/* end of falconinstance.h */
